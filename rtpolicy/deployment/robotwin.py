"""Policy entry points for the unchanged RoboTwin2 evaluation interface."""
import importlib
import os


def _native_adapter():
    # RoboTwin's evaluator puts its policy directory on sys.path.
    return importlib.import_module('RTPolicy.deploy_policy')


def get_model(args):
    backend = args.get('backend', 'pytorch')
    requested_nfe = args.get('num_inference_timesteps')
    environment_nfe = os.environ.get('RTPOLICY_NFE')
    if requested_nfe is not None and environment_nfe and int(requested_nfe) != int(environment_nfe):
        raise ValueError('Configuration NFE and RTPOLICY_NFE must match')
    requested_nfe = requested_nfe if requested_nfe is not None else environment_nfe
    if backend == 'pytorch':
        if not isinstance(args.get('ckpt_setting'), str) or not args['ckpt_setting']:
            raise ValueError('Set ckpt_setting to the checkpoint directory')
        runner = _native_adapter().get_model(args)
        if requested_nfe is not None:
            runner.policy.num_inference_timesteps = int(requested_nfe)
        return runner
    if backend not in ('onnx', 'tensorrt'):
        raise ValueError(f'Unknown inference backend: {backend}')
    if not args.get('model_path'):
        raise ValueError(f'{backend} requires model_path')
    from .compiled import CompiledPolicyRunner
    runner = CompiledPolicyRunner(args['model_path'], backend,
                                  metadata_path=args.get('metadata_path'))
    if requested_nfe and int(requested_nfe) != runner.num_inference_timesteps:
        raise ValueError(f'RTPOLICY_NFE={requested_nfe} differs from exported NFE={runner.num_inference_timesteps}')
    print(f'[RTPolicy] backend={backend}, NFE={runner.num_inference_timesteps}')
    return runner


def eval(task_env, model, observation):
    return _native_adapter().eval(task_env, model, observation)


def reset_model(model):
    return _native_adapter().reset_model(model)
