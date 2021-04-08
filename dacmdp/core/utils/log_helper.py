from PIL import Image

def write_gif(episode_images, media_path, save_mp4=True):
    gif_path, mp4_path = media_path + '.gif', media_path + '.mp4'
    episode_images = [Image.fromarray(_img) for _img in episode_images]

    # save as gif
    episode_images[0].save(gif_path, save_all=True, append_images=episode_images[1:], optimize=False, loop=1)

    # save video
    if save_mp4:
        import moviepy.editor as mp
        clip = mp.VideoFileClip(gif_path)
        clip.write_videofile(mp4_path)