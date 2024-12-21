
# models.py
import os
from PIL import Image
import torch
import numpy as np
import imageio
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, DPMSolverMultistepScheduler

# Load models
image_pipe = StableDiffusionXLPipeline.from_pretrained(
    "segmind/SSD-1B",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

video_pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16"
)
video_pipe.scheduler = DPMSolverMultistepScheduler.from_config(video_pipe.scheduler.config)
video_pipe.enable_model_cpu_offload()
video_pipe.enable_vae_slicing()

def generate_image(prompt, user_id):
    user_dir = os.path.join("generated_content", user_id)  # Updated path
    os.makedirs(user_dir, exist_ok=True)
    image_paths = []
    image_urls = []

    try:
        # Generate images
        with torch.no_grad():
            images = image_pipe(prompt, num_images_per_prompt=2, num_inference_steps=10).images

        for i, img in enumerate(images):
            img_path = os.path.join(user_dir, f"image_{i+1}.png")
            img.save(img_path)
            image_paths.append(img_path)
            image_urls.append(f"/generated_content/{user_id}/image_{i+1}.png")  # Updated URL path

    finally:
        # Free up GPU memory
        torch.cuda.empty_cache()

    return images, image_urls


def generate_video(prompt, user_id, num_videos=2):
    user_dir = os.path.join("generated_content", user_id)  # Updated path
    os.makedirs(user_dir, exist_ok=True)

    video_urls = []
    video_frames_list = []

    # Generate videos one by one (sequentially) for debugging
    for index in range(num_videos):
        video_path = os.path.join(user_dir, f"video_{index+1}.mp4")

        try:
            # Generate video frames for each video (sequential)
            with torch.no_grad():
                video_frames = video_pipe(prompt, num_inference_steps=3).frames

            rgb_frames = [np.array(Image.fromarray(frame).convert("RGB")) for frame in video_frames]

            # Save video
            with imageio.get_writer(video_path, fps=10) as writer:
                for frame in rgb_frames:
                    writer.append_data(frame)

            # Add the video URL
            video_urls.append(f"/generated_content/{user_id}/video_{index+1}.mp4")  # Updated URL path
            video_frames_list.append(video_frames)

        except Exception as e:
            print(f"Error generating video {index+1}: {e}")

        finally:
            # Free up GPU memory after each video is processed
            torch.cuda.empty_cache()

    return video_frames_list, video_urls






# import os
# from PIL import Image
# import torch
# import numpy as np
# import imageio
# from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, DPMSolverMultistepScheduler

# # Determine whether to use CUDA or CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load models
# image_pipe = StableDiffusionXLPipeline.from_pretrained(
#     "segmind/SSD-1B",
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16"
# ).to(device)  # Move model to CUDA or CPU

# video_pipe = DiffusionPipeline.from_pretrained(
#     "damo-vilab/text-to-video-ms-1.7b",
#     torch_dtype=torch.float16,
#     variant="fp16"
# ).to(device)  # Move model to CUDA or CPU
# video_pipe.scheduler = DPMSolverMultistepScheduler.from_config(video_pipe.scheduler.config)
# video_pipe.enable_model_cpu_offload()
# video_pipe.enable_vae_slicing()

# def generate_image(prompt, user_id):
#     user_dir = os.path.join("generated_content", user_id)  # Updated path
#     os.makedirs(user_dir, exist_ok=True)
#     image_paths = []
#     image_urls = []

#     try:
#         # Generate images
#         with torch.no_grad():
#             images = image_pipe(prompt, num_images_per_prompt=2, num_inference_steps=10).images

#         for i, img in enumerate(images):
#             img_path = os.path.join(user_dir, f"image_{i+1}.png")
#             img.save(img_path)
#             image_paths.append(img_path)
#             image_urls.append(f"/generated_content/{user_id}/image_{i+1}.png")  # Updated URL path

#     finally:
#         # Free up GPU memory
#         torch.cuda.empty_cache()

#     return images, image_urls


# def generate_video(prompt, user_id, num_videos=2):
#     user_dir = os.path.join("generated_content", user_id)  # Updated path
#     os.makedirs(user_dir, exist_ok=True)

#     video_urls = []
#     video_frames_list = []

#     # Generate videos one by one (sequentially) for debugging
#     for index in range(num_videos):
#         video_path = os.path.join(user_dir, f"video_{index+1}.mp4")

#         try:
#             # Generate video frames for each video (sequential)
#             with torch.no_grad():
#                 video_frames = video_pipe(prompt, num_inference_steps=3).frames

#             rgb_frames = [np.array(Image.fromarray(frame).convert("RGB")) for frame in video_frames]

#             # Save video
#             with imageio.get_writer(video_path, fps=10) as writer:
#                 for frame in rgb_frames:
#                     writer.append_data(frame)

#             # Add the video URL
#             video_urls.append(f"/generated_content/{user_id}/video_{index+1}.mp4")  # Updated URL path
#             video_frames_list.append(video_frames)

#         except Exception as e:
#             print(f"Error generating video {index+1}: {e}")

#         finally:
#             # Free up GPU memory after each video is processed
#             torch.cuda.empty_cache()

#     return video_frames_list, video_urls

