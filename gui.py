import dearpygui.dearpygui as dpg
import torch
import numpy as np
import time
import run
import os
from PIL import Image

STATE = {
    "rvk": None,

    "fg_kmeans": None,
    "fg": None,
    "valid_idx": None,

    "og_density_grid": None,
    "prev_mask": None,

    "positive_buffer": None,
    "negative_buffer": None,

    "rgbs": {},
    "features": {},

    "pixel_values": set(),
    "stroke_index": None,

    "stroke_color": np.array([0.0, 1.0, 0.0, 1.0])
}

def breakpoint_callback(sender, app_data):
    breakpoint()

def save_masks_callback(sender, app_data):
    dpg.set_value("save mask text", "Generating Masks...")
    os.makedirs("masks", exist_ok=True)
    os.makedirs("masks/train", exist_ok=True)
    os.makedirs("masks/test", exist_ok=True)
    alpha_threshold = dpg.get_value("alpha threshold")
    masks = run.get_masks(STATE["rvk"], STATE["cfg"], STATE["data_dict"], "train", alpha_threshold)
    for i, mask in enumerate(masks):
        Image.fromarray((mask * 255).astype(np.uint8)).save(f'masks/train/{i}.png')
    masks = run.get_masks(STATE["rvk"], STATE["cfg"], STATE["data_dict"], "test", alpha_threshold)
    for i, mask in enumerate(masks):
        Image.fromarray((mask * 255).astype(np.uint8)).save(f'masks/test/{i}.png')
    dpg.set_value("save mask text", "Done!")

def save_model_callback(sender, app_data):
    dpg.set_value("save text", "Saving Model.")
    osd = torch.load(os.path.join(STATE["cfg"].basedir, STATE["cfg"].expname, "fine_last.tar"))['optimizer_state_dict']
    with torch.no_grad():
        torch.save({
            'global_step': STATE["start"],
            'model_kwargs': STATE['rvk']['model'].get_kwargs(),
            'model_state_dict': STATE['rvk']['model'].state_dict(),
            'optimizer_state_dict': osd,
        }, os.path.join(STATE["cfg"].basedir, STATE["cfg"].expname, dpg.get_value("save name")))
    dpg.set_value("save text", "Saved Model.")
    torch.cuda.empty_cache()

def display_original_callback(sender, app_data):
    with torch.no_grad():
        mask = torch.ones(STATE["rvk"]["model"].segmentation_mask.shape, dtype=torch.float32).cuda()
        STATE["prev_mask"] = STATE["rvk"]["model"].segmentation_mask.cpu()
        STATE["rvk"]["model"].segmentation_mask = torch.nn.Parameter(mask, requires_grad=False)
        STATE["rvk"]["model"].density.grid = torch.nn.Parameter(STATE["og_density_grid"].cuda() * mask.cuda())

    STATE["rgbs"].clear()
    img_idx = dpg.get_value("idx slider")
    change_image_callback(None, img_idx)
    torch.cuda.empty_cache()

def clear_strokes_callback(sender, app_data):
    STATE["stroke_index"][:] = 0
    STATE["rgbs"].clear()
    img_idx = dpg.get_value("idx slider")
    change_image_callback(None, img_idx)

def replace_positive_buffer_callback(sender, app_data):
    STATE["positive_buffer"] = STATE["rvk"]["model"].segmentation_mask.clone().cpu()
    torch.cuda.empty_cache()

def add_to_positive_buffer_callback(sender, app_data):
    STATE["positive_buffer"] += STATE["rvk"]["model"].segmentation_mask.clone().cpu().float()
    STATE["positive_buffer"] = torch.clamp(STATE["positive_buffer"], max=1.0)
    torch.cuda.empty_cache()

def display_positive_buffer_callback(sender, app_data):
    with torch.no_grad():
        mask = STATE["positive_buffer"]
        STATE["prev_mask"] = STATE["rvk"]["model"].segmentation_mask.cpu()
        STATE["rvk"]["model"].segmentation_mask = torch.nn.Parameter(mask, requires_grad=False)
        STATE["rvk"]["model"].density.grid = torch.nn.Parameter(STATE["og_density_grid"].cuda() * mask.cuda())

    STATE["rgbs"].clear()
    img_idx = dpg.get_value("idx slider")
    change_image_callback(None, img_idx)

def replace_negative_buffer_callback(sender, app_data):
    STATE["negative_buffer"] = STATE["rvk"]["model"].segmentation_mask.clone().cpu()

def add_to_negative_buffer_callback(sender, app_data):
    STATE["negative_buffer"] += STATE["rvk"]["model"].segmentation_mask.clone().cpu()
    STATE["negative_buffer"] = torch.clamp(STATE["negative_buffer"], max=1.0)

def display_negative_buffer_callback(sender, app_data):
    with torch.no_grad():
        mask = STATE["negative_buffer"]
        STATE["prev_mask"] = STATE["rvk"]["model"].segmentation_mask.cpu()
        STATE["rvk"]["model"].segmentation_mask = torch.nn.Parameter(mask, requires_grad=False)
        STATE["rvk"]["model"].density.grid = torch.nn.Parameter(STATE["og_density_grid"] * mask)

    STATE["rgbs"].clear()
    img_idx = dpg.get_value("idx slider")
    change_image_callback(None, img_idx)

def merge_buffers_callback(sender, app_data):
    with torch.no_grad():
        mask = (STATE["positive_buffer"].bool() & ~STATE["negative_buffer"].bool()).float()

        STATE["prev_mask"] = STATE["rvk"]["model"].segmentation_mask.cpu()
        STATE["rvk"]["model"].segmentation_mask = torch.nn.Parameter(mask, requires_grad=False)
        STATE["rvk"]["model"].density.grid = torch.nn.Parameter(STATE["og_density_grid"] * mask)

    STATE["rgbs"].clear()
    img_idx = dpg.get_value("idx slider")
    change_image_callback(None, img_idx)

def stroke_color_callback(sender, app_data):
    if app_data == "positive":
        STATE["stroke_color"] = np.array([0.0, 1.0, 0.0, 1.0])
    elif app_data == "negative":
        STATE["stroke_color"] = np.array([1.0, 0.0, 0.0, 1.0])

    # change existing color
    img_data = np.array(dpg.get_value("image"))
    img_data = img_data.reshape(STATE["height"], STATE["width"], STATE["channels"])
    idx_red = (img_data == np.array([0.0, 1.0, 0.0, 1.0]).reshape(1, 1, 4)).sum(-1) == 4
    idx_green = (img_data == np.array([1.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4)).sum(-1) == 4
    img_data[idx_red | idx_green] = STATE["stroke_color"]

    img_data = img_data.reshape(STATE["height"], STATE["width"], STATE["channels"])
    dpg.set_value("image", img_data)

def clear_pixels_callback(sender, app_data):
    STATE["stroke_index"][:] = 0
    dpg.set_value("pixels selected", 0)
    idx = dpg.get_value("idx slider")
    change_image_callback(None, idx)

def undo_callback(sender, app_data):
    with torch.no_grad():
        og_density_grid = STATE["og_density_grid"].cuda()
        prev_mask = STATE["prev_mask"].cuda()
        density = og_density_grid * prev_mask

        STATE["rvk"]["model"].segmentation_mask = torch.nn.Parameter(prev_mask, requires_grad=False)
        STATE["rvk"]["model"].density.grid = torch.nn.Parameter(density)

        img_idx = dpg.get_value("idx slider")
    STATE["rgbs"].clear()
    change_image_callback(None, img_idx)

def update_alpha_threshold(sender, app_data):
    STATE["rgbs"].clear()
    img_idx = dpg.get_value("idx slider")
    change_image_callback(None, img_idx)

def change_image_callback(sender, app_data):
    idx = app_data
    if idx not in STATE["rgbs"]:
        alpha_threshold = dpg.get_value("alpha threshold")
        rgbs, features = run.render_single_image(STATE["rvk"], STATE["cfg"], STATE["data_dict"], idx, True, alpha_threshold)
        rgbs, features = rgbs.squeeze(0), features.squeeze(0)
        rgbs = np.concatenate([rgbs, np.ones_like(rgbs[:,:,0:1])], -1)
        STATE["rgbs"][idx] = rgbs.transpose((0, 1, 2))
        STATE["features"][idx] = features[:, :, :STATE["dino_dim"]]
    
    STATE["rgbs"][idx][STATE["stroke_index"][idx]] = STATE["stroke_color"]
    dpg.set_value("image", STATE["rgbs"][idx].reshape(-1))

def mouse_down_handler(sender, app_data):
    if dpg.is_item_hovered("main"):
        x, y = dpg.get_mouse_pos()
        x, y = int(x - 10), int(y - 10)
        STATE["pixel_values"].add((y, x))

def mouse_release_handler(sender, app_data):
    if dpg.is_item_hovered("main"):
        # get image in the current buffer
        img_data = np.array(dpg.get_value("image"))
        img_data = img_data.reshape(STATE["height"], STATE["width"], STATE["channels"])

        # get the values from the current set
        pixels = np.array(list(STATE["pixel_values"]), dtype=int)
        # remove invalid indices
        index = ~((pixels[:, 0] >= STATE["height"]) | (pixels[:, 0] < 0) | (pixels[:, 1] >= STATE["width"]) | (pixels[:, 1] < 0))
        pixels = pixels[index]

        # TODO: Change this form squre to circular strokes using faster OpenCV implementation
        idx = dpg.get_value("idx slider")
        for x in range(-3, 4):
            for y in range(-3, 4):
                img_data[pixels[:,0] + x, pixels[:,1] + y] = STATE["stroke_color"]
                STATE["stroke_index"][idx, pixels[:, 0] + x, pixels[:, 1] + y] = 1

        dpg.set_value("image", img_data.reshape(-1))

    # clear the set
    dpg.set_value("pixels selected", STATE["stroke_index"].sum())
    STATE["pixel_values"].clear()

def hcr_callback(sender, app_data):
    dpg.set_value("hcr text", "Getting HCR.")
    times = [time.time()]
    with torch.no_grad():
        img_data = np.array(dpg.get_value("image"))
        img_data = img_data.reshape(STATE["height"], STATE["width"], STATE["channels"])
        index = img_data == STATE["stroke_color"]
        index = (index.sum(-1) == 4)
        img_idx = dpg.get_value("idx slider")

        STATE["faiss_kmeans"] = run.do_kmeans_clustering_multiview(STATE["stroke_index"], STATE["features"])

        thresh = dpg.get_value("threshold")
        mask = run.query_kmeans_clustering(STATE["faiss_kmeans"], STATE["fg_kmeans"], STATE["valid_idx"], thresh, STATE["og_density_grid"].shape[2:])
        times.append(time.time())

        STATE["prev_mask"] = STATE["rvk"]["model"].segmentation_mask.cpu()
        STATE["rvk"]["model"].segmentation_mask = torch.nn.Parameter(mask, requires_grad=False)
        STATE["rvk"]["model"].density.grid = torch.nn.Parameter(STATE["rvk"]["model"].density.grid * mask)

        STATE["rgbs"].clear()
        STATE["stroke_index"][:] = 0

        change_image_callback(None, img_idx)
        times.append(time.time())
    torch.cuda.empty_cache()
    dpg.set_value("hcr text", f"Time taken to get HCR: {round(times[1]-times[0], 4)} Time taken to render: {round(times[2]-times[1], 4)}")

def region_grower_callback(sender, app_data):
    torch.cuda.empty_cache()
    dpg.set_value("region grower text", "Growing. Please don't spam.")
    times = [time.time()]
    with torch.no_grad():
        mask = STATE["rvk"]["model"].segmentation_mask.data
        sigma_d = dpg.get_value("sigma_d")
        sigma_f = dpg.get_value("sigma_f")
        mask = run.run_region_grower(mask, STATE["fg"], sigma_d, sigma_f)
        times.append(time.time())
        STATE["prev_mask"] = STATE["rvk"]["model"].segmentation_mask.cpu()
        STATE["rvk"]["model"].segmentation_mask = torch.nn.Parameter(mask, requires_grad=False)
        STATE["rvk"]["model"].density.grid = torch.nn.Parameter(STATE["og_density_grid"].cuda() * mask)
        STATE["rgbs"].clear()
        img_idx = dpg.get_value("idx slider")
        change_image_callback(None, img_idx)
        times.append(time.time())
    dpg.set_value("region grower text", f"Time taken to grow: {round(times[1]-times[0], 4)} Time taken to render: {round(times[2]-times[1], 4)}")

if __name__ == "__main__":
    with torch.no_grad():
        parser = run.config_parser()
        args = parser.parse_args()

        STATE["dino_dim"] = args.dino_dim
        STATE["cfg"], STATE["data_dict"], STATE["device"] = run.do_setup(args)
        STATE["rvk"], STATE["optimizer"], STATE["start"] = run.load_model(args, STATE["cfg"], STATE["data_dict"], STATE["device"])
        STATE["og_density_grid"] = STATE["rvk"]["model"].density.grid.clone().cpu()
        STATE["positive_buffer"] = STATE["rvk"]["model"].segmentation_mask.clone()
        STATE["negative_buffer"] = 1.0 - STATE["rvk"]["model"].segmentation_mask.clone()
        STATE["fg"], STATE["fg_kmeans"], STATE["valid_idx"] = run.reconstruct_feature_grid(STATE["rvk"], STATE["dino_dim"])
        STATE["height"], STATE["width"] = STATE["data_dict"]["HW"][0]
        STATE["height"], STATE["width"] = int(STATE["height"]), int(STATE["width"])
        STATE["channels"] = 4
        STATE["stroke_index"] = np.zeros([len(STATE["data_dict"]["HW"]), STATE["height"], STATE["width"]], dtype=bool)
        torch.cuda.empty_cache()


        dpg.create_context()
        dpg.create_viewport(title='ISRF', width=1800, height=1000)

        # handlers
        with dpg.handler_registry():
            dpg.add_mouse_down_handler(callback=mouse_down_handler)
            dpg.add_mouse_release_handler(callback=mouse_release_handler)

        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(width=STATE["width"], height=STATE["height"], default_value=np.zeros(STATE["width"] * STATE["height"] * 4, dtype=float), tag="image")

        with dpg.window(label="main", tag="main", no_resize=True, no_move=True, no_close=True):
            dpg.add_image("image")
        with dpg.window(label="control", tag="control", pos=[1000, 0], width=500, height=700, no_close=True):
            dpg.add_button(label="undo", tag="undo", callback=undo_callback)
            dpg.add_button(label="display original", tag="dislay original", callback=display_original_callback)
            dpg.add_button(label="clear strokes", tag="clear strokes", callback=clear_strokes_callback)

            dpg.add_slider_int(label="idx slider", tag="idx slider", min_value=0, max_value=len(STATE["data_dict"]["HW"])-1, clamped=True, callback=change_image_callback)
            dpg.add_slider_double(label="alpha threshold", tag="alpha threshold", min_value=0.0, max_value=0.2, default_value=0.0, clamped=True, callback=update_alpha_threshold, no_input=False)

            with dpg.collapsing_header(label="Get High Confidence Region"):
                with dpg.group(horizontal=True):
                    dpg.add_radio_button(items=["positive", "negative"], label="stroke color", tag="stroke color", callback=stroke_color_callback)
                    dpg.add_slider_double(label="threshold", tag="threshold", min_value=0.0, max_value=2.0, default_value=0.4)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="clear pixels", tag="clear pixels", callback=clear_pixels_callback)
                    dpg.add_text(label="pixels selected", tag="pixels selected", default_value="0")
                dpg.add_button(label="get hcr", tag="get hcr", callback=hcr_callback)
                dpg.add_text(label="hcr text", tag="hcr text", default_value="Not called yet!")

            with dpg.collapsing_header(label="Region Grower"):
                dpg.add_slider_double(label="sigma_d", tag="sigma_d", min_value=0.0, max_value=20.0, default_value=1.0)
                dpg.add_slider_double(label="sigma_f", tag="sigma_f", min_value=0.0, max_value=20.0, default_value=10.0)
                dpg.add_button(label="grow region", tag="grow region", callback=region_grower_callback)
                dpg.add_text(label="region grower text", tag="region grower text", default_value="Not called yet!")

            with dpg.collapsing_header(label="Buffer Management"):
                with dpg.group(horizontal=True, label="POSITIVE"):
                    dpg.add_button(label="replace positive buffer", tag="replace positive buffer", callback=replace_positive_buffer_callback)
                    dpg.add_button(label="add to positive buffer", tag="add to positive buffer", callback=add_to_positive_buffer_callback)
                    dpg.add_button(label="display positive buffer", tag="display positive buffer", callback=display_positive_buffer_callback)
                with dpg.group(horizontal=True, label="NEGATIVE"):
                    dpg.add_button(label="replace negative buffer", tag="replace negative buffer", callback=replace_negative_buffer_callback)
                    dpg.add_button(label="add to negative buffer", tag="add to negative buffer", callback=add_to_negative_buffer_callback)
                    dpg.add_button(label="display negative buffer", tag="display negative buffer", callback=display_negative_buffer_callback)
                dpg.add_button(label="merge buffers", tag="merge buffers", callback=merge_buffers_callback)
            
            with dpg.collapsing_header(label="Save interface"):
                dpg.add_text(label="save text", tag="save text", default_value="Nothing saved yet!")
                dpg.add_input_text(tag="save name", default_value="segmented.tar")
                dpg.add_button(label="save model", tag="save model", callback=save_model_callback)
                dpg.add_button(label="save masks", tag="save masks", callback=save_masks_callback)
                dpg.add_text(label="save mask text", tag="save mask text", default_value="Nothing saved yet!")


            # dpg.add_button(label="breakpoint", tag="breakpoint", callback=breakpoint_callback)

        # dpg.show_debug()
        # dpg.show_item_registry()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        # dpg.set_primary_window("isrf main window", True)

        img_idx = dpg.get_value("idx slider")
        change_image_callback(None, img_idx)
        display_original_callback(None, None)
        torch.cuda.empty_cache()

        dpg.start_dearpygui()
        dpg.destroy_context()