

def write_gif(episode_images, metrics, metric_metas, gif_path, scaling = 0.75,  save_mp4=True):
    """
    metrics = list of metric array 
    labels = list of x and y lavels for each metric array
    """
    episode_len = len(episode_images)
    assert all([len(episode_images) == len(m) for m in metrics])
    
    import plotly.graph_objects as go
    from io import BytesIO
    from PIL import Image
    import numpy as np
    
    rep_figs, rep_imgs= [], [None]*len(metrics)
    for i, m, meta in zip(range(len(metrics)), metrics, metric_metas): 
        rep_figs.append(go.Figure(data=go.Scatter(x=[], y=[])))
        rep_figs[-1].update_layout(
            title=meta["title"],
            xaxis_title=meta["xaxis_title"],
            yaxis_title=meta["yaxis_title"],
        )

    episode_stats = []
    _obs = Image.fromarray(episode_images[0])
    width, height = int(scaling * _obs.width), int(scaling * _obs.height)
    step_i = 0


    while step_i < len(episode_images):
        # update figure for all metrics
        for i, m, meta in zip(range(len(metrics)), metrics, metric_metas):
            rep_figs[i]['data'][0]['x'] += tuple([step_i])
            rep_figs[i]['data'][0]['y'] += tuple([m[step_i]])

            rep_imgs[i] = Image.open(BytesIO(rep_figs[i].to_image(format="png",
                                 width=width, height=height)))

        # obs
        obs = Image.fromarray(episode_images[step_i])
        obs = obs.resize((width, height), Image.ANTIALIAS)

        # combine repeat image + actual obs + score image
        all_widths = [obs.width , *[img.width for img in rep_imgs]]
        overall_width = np.sum(all_widths)
        overall_img = Image.new('RGB', (overall_width, height))
        overall_img.paste(obs, (0, 0))
        rep_img_start_positions = np.cumsum(all_widths)
                              
        for rep_img, pos in zip(rep_imgs,rep_img_start_positions[:-1]):                     
            overall_img.paste(rep_img, (pos, 0))
            episode_stats.append(overall_img)

        # incr counters
        step_i += 1

    assert step_i == len(episode_images)

    # save as gif
    episode_stats[0].save(gif_path, save_all=True, append_images=episode_stats[1:], optimize=False, loop=1)

    # save video
    if save_mp4:
        import moviepy.editor as mp
        clip = mp.VideoFileClip(gif_path)
        clip.write_videofile(gif_path.replace('.gif', '.mp4'))

    return gif_path