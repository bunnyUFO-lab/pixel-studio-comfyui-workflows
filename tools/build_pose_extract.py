"""Generates workflows/pose/extract-pose.json.

Edit the constants below and re-run:  python tools/build_pose_extract.py

Picture or clip -> pose skeletons: VHS loads the clip (a GIF, an MP4, or a one-frame GIF of a picture), SDPose
(core ComfyUI nodes, Comfy-Org/SDPose whole-body model) finds the body, hand and foot keypoints of each frame,
SDPoseDrawKeypoints draws them OpenPose-style, and the keypoints are printed as JSON for the app (Sprite Studio
keeps them with the reference and draws its own skeletons from them, scaled to the sprite, for AI Animate).
"""
from comfy_workflow import BLUE, GREEN, WORKFLOWS, YELLOW, Graph, inp, out

OUT = WORKFLOWS / "pose" / "extract-pose.json"

SDPOSE = "sdpose_wholebody_fp16.safetensors"
SDPOSE_URL = "https://huggingface.co/Comfy-Org/SDPose/resolve/main/checkpoints/" + SDPOSE
FRAMES = 16                       # frames read from the clip (0 = all)
BATCH = 8

NOTE = """Picture or clip -> pose skeletons (SDPose)

1. Clip (blue): a GIF or video of a figure (a picture works as a one-frame GIF). frame_load_cap = how many frames,
   select_every_nth to skip frames.
2. SDPose finds the whole-body keypoints (body, hands, feet, face) of the main figure in each frame.
3. 'Pose images' = the skeletons drawn OpenPose-style (the same kind of picture pose-guided video models such as
   Wan VACE follow); 'Keypoints' = the same as JSON.
Model: Comfy-Org/SDPose sdpose_wholebody_fp16 (1.9 GB) in models/checkpoints."""

g = Graph()
node, link = g.node, g.link

node(1, "CheckpointLoaderSimple", [0, 0], [380, 98], outputs=[out("MODEL", "MODEL"), out("CLIP", "CLIP"), out("VAE", "VAE")],
     widgets=[SDPOSE], title="SDPose model",
     props={"Node name for S&R": "CheckpointLoaderSimple", "models": [{"name": SDPOSE, "url": SDPOSE_URL, "directory": "checkpoints"}]})
node(2, "VHS_LoadVideo", [0, 160], [380, 420],
     inputs=[inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
     outputs=[out("IMAGE", "IMAGE"), out("frame_count", "INT"), out("audio", "AUDIO"), out("video_info", "VHS_VIDEOINFO")],
     widgets={"video": "example.gif", "force_rate": 0, "custom_width": 0, "custom_height": 0, "frame_load_cap": FRAMES,
              "skip_first_frames": 0, "select_every_nth": 1, "format": "AnimateDiff",
              "videopreview": {"hidden": False, "paused": False, "params": {}}},
     title="Clip", color=BLUE, props={"Node name for S&R": "VHS_LoadVideo", "cnr_id": "comfyui-videohelpersuite"})
node(3, "SDPoseKeypointExtractor", [420, 0], [340, 120],
     inputs=[inp("model", "MODEL"), inp("vae", "VAE"), inp("image", "IMAGE")], outputs=[out("POSE_KEYPOINT", "POSE_KEYPOINT")],
     widgets=[BATCH], title="Find keypoints", color=YELLOW)
# draw_body, draw_hands, draw_face, draw_feet, stick_width, face_point_size, score_threshold, draw_head
node(4, "SDPoseDrawKeypoints", [420, 180], [340, 260], inputs=[inp("keypoints", "POSE_KEYPOINT")], outputs=[out("IMAGE", "IMAGE")],
     widgets=[True, True, False, True, 4, 3, 0.3, True], title="Draw skeletons")
node(5, "SaveImage", [800, 180], [360, 360], inputs=[inp("images", "IMAGE")], widgets=["pose/skeleton"], title="Pose images",
     color=GREEN)
node(6, "PreviewAny", [800, 0], [360, 140], inputs=[inp("source", "*")], title="Keypoints")
node(7, "Note", [0, 640], [520, 260], widgets=[NOTE], title="READ ME", props={})

link(1, 0, 3, 0)
link(1, 2, 3, 1)
link(2, 0, 3, 2)
link(3, 0, 4, 0)
link(4, 0, 5, 0)
link(3, 0, 6, 0)

g.save(OUT, "pose-extract-pose", groups=[("Pose skeletons (SDPose)", [-20, -60, 1200, 640], "#3f789e")], scale=0.75)
