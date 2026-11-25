"""Serve pi05_droid_jointpos with OpenPI. Launch inside the OpenPI environment; does not modify upstream configs."""
import argparse
import dataclasses
import logging
import os

import numpy as np
from unittest import mock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--restore-concurrent-gb", type=int, default=4, help="Limit checkpoint read staging to fit hosts with little system RAM")
    args = parser.parse_args()
    if args.restore_concurrent_gb <= 0:
        parser.error("--restore-concurrent-gb must be positive")
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.serving import websocket_policy_server
    from openpi import transforms
    from openpi.policies import droid_policy
    import orbax.checkpoint as ocp

    checkpoint = "gs://openpi-assets/checkpoints/pi05_droid_jointpos"
    # Jointpos uses the same architecture / DROID IO and its OWN normalization assets.
    cfg = config.get_config("pi05_droid")
    mask = transforms.make_bool_mask(7, -1)
    data = dataclasses.replace(cfg.data, data_transforms=lambda model: transforms.Group(
        inputs=[droid_policy.DroidInputs(model_type=model.model_type), transforms.DeltaActions(mask)],
        outputs=[transforms.AbsoluteActions(mask), droid_policy.DroidOutputs()],
    ))
    cfg = dataclasses.replace(cfg, data=data, policy_metadata={**(cfg.policy_metadata or {}),
                              "checkpoint": checkpoint, "action_space": "absolute_joint_position",
                              "restore_concurrent_gb": args.restore_concurrent_gb,
                              "jax_allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default")})
    # Upstream restore_params constructs an unbounded PyTreeCheckpointer. Bound
    # its temporary host read buffers while retaining the same restore arguments,
    # bfloat16 weights, model, and normalization assets. Restore the factory as
    # soon as this single startup load is complete.
    def bounded_checkpointer():
        return ocp.Checkpointer(ocp.PyTreeCheckpointHandler(restore_concurrent_gb=args.restore_concurrent_gb))

    with mock.patch.object(ocp, "PyTreeCheckpointer", bounded_checkpointer):
        policy = policy_config.create_trained_policy(cfg, checkpoint)
    # Common random numbers for paired simulator experiments. Keep the ordinary
    # inference path (and its server-side RNG sequence) available to clients.
    horizon = cfg.model.action_horizon
    dim = cfg.model.action_dim

    class ExperimentPolicy:
        def __init__(self):
            self.inference_id = 0

        def infer(self, observation):
            if "sampling_seed" not in observation:
                reply = policy.infer(observation)
            else:
                request = dict(observation)
                seed = int(request.pop("sampling_seed"))
                noise = np.random.default_rng(seed).standard_normal((horizon, dim), dtype=np.float32)
                reply = policy.infer(request, noise=noise)
            self.inference_id += 1
            return {**reply, "inference_id": self.inference_id}

    websocket_policy_server.WebsocketPolicyServer(ExperimentPolicy(), host="0.0.0.0", port=args.port,
                                                  metadata={**policy.metadata, "sampling_seed": True,
                                                            "prediction_horizon": horizon,
                                                            "fresh_chunk_inference": True}).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
