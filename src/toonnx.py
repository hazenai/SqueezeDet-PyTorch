import os
import torch
from model.squeezedet import SqueezeDetWithResolver
from torch.onnx import export
from logging import debug
from onnx import load, save
from onnxsim import simplify
from utils.misc import load_dataset
from utils.config import Config
from utils.model import load_model


def ToOnnx(cfg):

    dataset = load_dataset(cfg.dataset)('val', cfg)
    cfg = Config().update_dataset_info(cfg, dataset)
    Config().print(cfg)

    model = SqueezeDetWithResolver(cfg)
    if cfg.qat:

        model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        fused_model = copy.deepcopy(model)
        fused_model.fuse_model()
        assert model_equivalence(model_1=model, model_2=fused_model, device='cpu', rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,cfg.input_size[0],cfg.input_size[1])), "Fused model is not equivalent to the original model!"
        model = torch.quantization.prepare_qat(fused_model)

    model = load_model(model, cfg.load_model, cfg)
        

    inp = torch.rand([1,3,cfg.input_size[0], cfg.input_size[1]]).cuda()
    

    inputs = ['input']
    outputs = ['pred_boxes', 'pred_scores', 'pred_class_probs']

    dynamic_axes = {k: {0 : "batch_size"} for k in [*inputs,*outputs]}
    onnx_path = os.path.join(cfg.save_dir, cfg.load_model.split('/')[-1][:-4] + '.onnx')
    onnx_sim_path = os.path.join(cfg.save_dir, cfg.load_model.split('/')[-1][:-4] + '_sim.onnx')
    export(
        model.cuda(),
        inp,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=inputs,
        output_names=outputs,
        dynamic_axes=dynamic_axes
    )

    onnx_model = load(onnx_path)
    onnx_outputs = [node.name for node in onnx_model.graph.output]
    debug(f"ONNX ouputs: {onnx_outputs}")

    debug("Simplifying ONNX model")
    onnx_model_sim, check = simplify(
        onnx_model,
        input_shapes={"input": [1,3,cfg.input_size[0], cfg.input_size[1]]}
    )
    assert check, "Simplified ONNX model could not be validated"

    debug("Saving simplified ONNX model")
    save(onnx_model_sim, onnx_sim_path)