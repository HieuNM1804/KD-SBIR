import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from src import benchmark_inference_speed as bench


class SpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_collect_and_seeded_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for mode in ("sketch","photo"):
                p=root/mode/"cake"
                p.mkdir(parents=True)
                for i in range(5):
                    Image.new("RGB",(16,24),(i*40,50,80)).save(p/f"{i}.png")
                (p/"ignored.txt").write_text("not an image")
            with patch.dict(bench.UNSEEN_CLASSES,{"test":["cake"]}):
                paths=bench.collect(root,"test")
            self.assertEqual(len(paths["photo"]),5)
            a,pa=bench.sample_images(paths["sketch"],3,42)
            b,pb=bench.sample_images(paths["sketch"],3,42)
            self.assertEqual(pa,pb)
            torch.testing.assert_close(a,b)
            self.assertEqual(tuple(a.shape),(3,3,224,224))

    def test_measure_warmup_and_raw_samples(self):
        calls=[]
        measured=bench.measure(lambda:calls.append(1),torch.device("cpu"),2,5)
        self.assertEqual(len(calls),7)
        self.assertEqual(len(measured["samples_ms"]),5)
        self.assertGreaterEqual(measured["p95_ms"],measured["median_ms"])
        self.assertIsNone(measured["peak_allocated_mib"])
        rate=bench.throughput(lambda:None,4,torch.device("cpu"),2,5)
        self.assertGreater(rate["images_per_second"],0)

    def test_full_pipeline_real_and_synthetic_gallery(self):
        class FakeEncoder(torch.nn.Module):
            def forward(self,x,modality="sketch"):
                return x.mean(dim=(-1,-2))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for mode in ("sketch","photo"):
                p=root/"dataset"/mode/"cake"
                p.mkdir(parents=True)
                Image.new("RGB",(16,16),(100,160,210)).save(p/"one.png")
            config=root/"datasets.json"
            config.write_text(json.dumps({"quickdraw":str(root/"dataset")}))
            for gallery in ("real","synthetic"):
                with patch.dict(bench.UNSEEN_CLASSES,{"quickdraw":["cake"]}), patch.object(
                    bench,"load_encoder",return_value=(FakeEncoder(),{"name":"test-only"})):
                    result=bench.main(["--datasets-json",str(config),"--models-root",str(root),
                        "--models","clip_b32","--device","cpu","--precision","fp32",
                        "--batch-sizes","1","2","--warmup","1","--iterations","2",
                        "--sample-images","1","--gallery",gallery,"--output-dir",str(root/"results")])
                self.assertTrue(result["complete"])
                self.assertEqual(len(result["measurements"]),6)
                self.assertEqual(result["measurements"][-1]["gallery_photos"],1)
                self.assertEqual(result["measurements"][-1]["top_k"],1)
                self.assertEqual(result["measurements"][-1]["gallery_kind"],gallery)
            outputs=list((root/"results").glob("*/summary.json"))
            self.assertEqual(len(outputs),2)
            for path in outputs:
                self.assertTrue(path.with_name("measurements.csv").is_file())
                self.assertTrue(json.loads(path.read_text())["complete"])

    def test_clip_loader_rebuilds_original_visual_weights(self):
        from clip.model import CLIP
        from types import SimpleNamespace
        model=CLIP(embed_dim=16, image_resolution=16, vision_layers=1, vision_width=64,
                   vision_patch_size=4, context_length=4, vocab_size=50,
                   transformer_width=64, transformer_heads=1, transformer_layers=1).float().eval()
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/"ViT-B-32.pt").touch()
            with patch("torch.jit.load",return_value=model):
                encoder, info=bench.load_encoder("clip_b32",SimpleNamespace(models_root=folder,precision="fp32",n_ctx_visual=3,prompt_depth=1,seed=42),torch.device("cpu"))
            image=torch.randn(1,3,16,16)
            # Project build_model first casts weights to fp16; compare at its round-trip precision.
            from clip.model import build_model
            from src.speed_prompts import wrap_student
            reference=build_model(dict(model.state_dict())).float().visual
            reference=wrap_student(reference,"openai",SimpleNamespace(n_ctx_visual=3,prompt_depth=1,seed=42),torch.device("cpu"))
            torch.testing.assert_close(encoder(image),reference(image))
            self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
            self.assertEqual(info["vision_parameters"],sum(p.numel() for p in encoder.visual.parameters()))

    def test_hf_and_pruned_prompt_paths_keep_tokens_at_each_layer(self):
        from types import SimpleNamespace
        from src.speed_prompts import wrap_student
        from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig
        from src.tinyclip_vendor.model import CLIP as PrunedCLIP
        vision=CLIPVisionConfig(image_size=16,patch_size=4,hidden_size=64,
                               intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,
                               attention_dropout=0.0)
        text=CLIPTextConfig(vocab_size=50,hidden_size=64,intermediate_size=128,
                           num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=4)
        config=CLIPConfig(text_config=text.to_dict(),vision_config=vision.to_dict(),projection_dim=16)
        config._attn_implementation="eager"
        hf=CLIPModel(config).eval()
        pruned=PrunedCLIP(embed_dim=16,vision_cfg={"image_size":16,"layers":2,"width":64,"patch_size":4},
                          text_cfg={"context_length":4,"vocab_size":50,"width":64,"heads":1,"layers":1},
                          mask_image=False,mask_text=False).eval()
        args=SimpleNamespace(n_ctx_visual=3,prompt_depth=2,seed=42)
        image=torch.randn(2,3,16,16)
        for raw,kind in ((hf,"huggingface"),(pruned,"auto_pruned")):
            model=wrap_student(SimpleNamespace(model=raw),kind,args,torch.device("cpu"))
            layers=(hf.vision_model.encoder.layers if kind=="huggingface" else pruned.visual.transformer.resblocks)
            lengths=[]
            axis=1 if kind=="huggingface" else 0
            hooks=[layer.register_forward_pre_hook(lambda module,inputs: lengths.append(inputs[0].shape[axis])) for layer in layers]
            with torch.inference_mode():
                a=model(image,"sketch")
                b=model(image,"photo")
            for hook in hooks:
                hook.remove()
            self.assertEqual(lengths,[20,20,20,20])  # 16 patches + CLS + 3 prompt tokens
            self.assertEqual(tuple(a.shape),(2,16))
            torch.testing.assert_close(a.norm(dim=-1),torch.ones(2))
            self.assertFalse(torch.allclose(a,b))
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertEqual(model.sketch_visual_prompt.ctx.dtype,torch.float32)
            # Bypass prompts: the copied visual wrapper must preserve official raw encoder output.
            with torch.inference_mode():
                actual=model.visual(image)
                expected=(hf.visual_projection(hf.vision_model(image).pooler_output) if kind=="huggingface"
                          else pruned.encode_image(image,normalized=False))
            torch.testing.assert_close(actual,expected)
            if torch.cuda.is_available():
                model.visual.to(device="cuda",dtype=torch.float16)
                model.to("cuda")
                with torch.inference_mode():
                    result=model(image.to(device="cuda",dtype=torch.float16),"sketch")
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(model.sketch_visual_prompt.ctx.dtype,torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available locally")
    def test_cuda_fp16_backbone_fp32_prompts(self):
        from types import SimpleNamespace
        from clip.model import CLIP, convert_weights
        from src.speed_prompts import wrap_student
        raw=CLIP(embed_dim=16,image_resolution=16,vision_layers=2,vision_width=64,
                 vision_patch_size=4,context_length=4,vocab_size=50,
                 transformer_width=64,transformer_heads=1,transformer_layers=1).visual
        convert_weights(raw)
        raw=raw.cuda()
        model=wrap_student(raw,"openai",SimpleNamespace(n_ctx_visual=3,prompt_depth=2,seed=42),torch.device("cuda"))
        images=torch.randn(1,3,16,16,device="cuda",dtype=torch.float16)
        with torch.inference_mode():
            features=model(images,"sketch")
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(model.sketch_visual_prompt.ctx.dtype,torch.float32)
        self.assertEqual(model.visual.conv1.weight.dtype,torch.float16)
        values=bench.measure(lambda:model(images,"sketch"),torch.device("cuda"),2,3)
        self.assertGreater(values["median_ms"],0)
        self.assertGreater(values["peak_allocated_mib"],0)
        self.assertEqual(len(values["samples_ms"]),3)

    def test_failure_is_saved_and_not_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            config=root/"datasets.json"
            config.write_text(json.dumps({"quickdraw":str(root)}))
            with patch.object(bench,"collect",return_value={"sketch":[str(root/"s.png")],"photo":[str(root/"p.png")]}), patch.object(
                bench,"sample_images",return_value=(torch.ones(1,3,224,224),[str(root/"s.png")])), patch.object(
                bench,"load_encoder",side_effect=FileNotFoundError("missing weights")):
                with self.assertRaisesRegex(RuntimeError,"Some models failed"):
                    bench.main(["--datasets-json",str(config),"--models-root",str(root),"--models","clip_b32",
                                "--device","cpu","--precision","fp32","--output-dir",str(root/"out")])
            report=json.loads(next((root/"out").glob("*/summary.json")).read_text())
            self.assertFalse(report["complete"])
            self.assertIn("missing weights",report["errors"][0]["error"])


if __name__=="__main__":
    unittest.main()
