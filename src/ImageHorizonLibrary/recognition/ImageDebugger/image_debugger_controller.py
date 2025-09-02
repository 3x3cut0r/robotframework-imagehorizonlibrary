"""Controller layer for the image debugger application."""

from .image_debugger_model import UILocatorModel
from .image_debugger_view import UILocatorView
from .template_matching_strategies import Pyautogui, Cv2
from .image_manipulation import ImageContainer, ImageFormat
from pathlib import Path
import os, glob
import pyperclip
import numpy as np
import webbrowser
from robot.api import logger as LOGGER


class UILocatorController:
    """Connects debugger model and view components."""

    def __init__(self, image_horizon_instance):
        """Create controller and associated model/view objects."""
        self.image_container = ImageContainer()
        self.model = UILocatorModel()
        self.image_horizon_instance = image_horizon_instance
        self.view = UILocatorView(self, self.image_container, self.image_horizon_instance)

    def main(self):
        """Run the main UI loop."""
        self.init_view()
        self.view.main()

    def init_view(self):
        """Initialise default values for the view widgets."""
        self.view.ref_dir_path.set(self.image_horizon_instance.reference_folder)
        self.view.scale_conf_lvl_ag.set(0.99)
        self.view.scale_sigma_edge.set(1.0)
        self.view.scale_low_thres_edge.set(0.1)
        self.view.scale_high_thres_edge.set(0.3)
        self.view.scale_conf_lvl_edge.set(0.8)
        self.view.matches_found.set("None")
        self.view.btn_edge_detec_debugger["state"] = "disabled"
        self.view.btn_run_pyautogui["state"] = "disabled"
        self.view.btn_run_edge["state"] = "disabled"
        self.view.btn_copy_strategy_snippet["state"] = "disabled"
        self.view.hint_msg.set("Ready")
        self.view.processing_done = False

    def help(self):
        """Open online keyword documentation."""
        webbrowser.open("https://eficode.github.io/robotframework-imagehorizonlibrary/doc/ImageHorizonLibrary.html")

    def load_needle_image_names(self, combobox=None):
        """Populate combobox with available reference images."""
        list_needle_images_names = []
        pattern = os.path.join(self.image_horizon_instance.reference_folder, "*.png")
        for needle_img_name in glob.glob(pattern):
            list_needle_images_names.append(os.path.basename(needle_img_name))
        if combobox:
            self.combobox = combobox
        self.combobox['values']=list_needle_images_names
        self.combobox.set('__ __ __ Select a reference image __ __ __')

    def reset_images(self):
        """Reset image placeholders in the view."""
        self.view.canvas_ref_img.itemconfig(
            self.view.ref_img,
            image=self.view.needle_img_blank,
        )
        self.view.canvas_desktop_img.itemconfig(
            self.view.desktop_img,
            image=self.view.haystack_img_blank,
        )

    def reset_results(self):
        """Clear previous recognition results from the UI."""
        self.view.btn_copy_strategy_snippet["state"] = "disabled"
        self.view.btn_edge_detec_debugger["state"] = "disabled"
        self.view.matches_found.set("None")
        self.view.label_matches_found.config(fg='black')
        self.view.set_strategy_snippet.set("")
    
    def refresh(self):
        """Reload available images and reset the UI state."""
        self.load_needle_image_names()
        self.view.btn_run_pyautogui["state"] = "disabled"
        self.view.btn_run_edge["state"] = "disabled"
        self.reset_results()
        self.reset_images()
        self.view._ready()
        self.view.processing_done=False

    def copy_to_clipboard(self):
        """Copy the current strategy snippet to the clipboard."""
        pyperclip.copy(self.strategy_snippet)

    def on_select(self, event):
        """Handle selection of a reference image from the combobox."""
        ref_image_name = self.view.combobox_needle_img_name.get()
        ref_image = Path(self.image_horizon_instance.reference_folder) / ref_image_name
        self.image_container.save_to_img_container(img=ref_image.__str__())
        self.needle_img = self.image_container.get_needle_image(ImageFormat.IMAGETK)
        self.view.canvas_ref_img.itemconfig(
            self.view.ref_img,
            image=self.needle_img,
        )

        self.view.btn_run_pyautogui["state"] = "normal"
        self.view.btn_run_edge["state"] = "normal"

    def _take_screenshot(self):
        """Capture a screenshot of the current desktop.

        The debugger window is minimised before taking the screenshot. To
        avoid leaving the window hidden in case of failures, a safety timer
        restores it after 10 seconds.
        """
        previous_state = self.view.state()

        try:
            self.view.iconify()
        except Exception as exc:  # pragma: no cover - depends on tk implementation
            raise RuntimeError(f"Failed to minimise debugger window: {exc}") from exc

        restore_id = self.view.after(10000, self.view.deiconify)
        try:
            return self.model.capture_desktop()
        finally:
            self.view.after_cancel(restore_id)
            try:
                self.view.deiconify()
                self.view.state(previous_state)
                self.view.lift()
            except Exception as exc:  # pragma: no cover - depends on tk implementation
                LOGGER.error(f"Restoring debugger window failed: {exc}")
    
    def on_click_run_default_strategy(self):
        """Execute recognition using the default strategy."""
        try:
            self.image_horizon_instance.set_strategy('default')
            self.image_horizon_instance.confidence = float(self.view.scale_conf_lvl_ag.get())
            self.image_container.save_to_img_container(
                self._take_screenshot(), is_haystack_img=True
            )

            matcher = Pyautogui(self.image_container, self.image_horizon_instance)
            self.coord = matcher.find_num_of_matches()
            matcher.highlight_matches()

            self.haystack_image = self.image_container.get_haystack_image(
                format=ImageFormat.IMAGETK
            )
            self.view.canvas_desktop_img.itemconfig(
                self.view.desktop_img, image=self.haystack_image
            )

            num_of_matches_found = len(self.coord)
            self.view.matches_found.set(num_of_matches_found)
            font_color = self.model.change_color_of_label(num_of_matches_found)
            self.view.label_matches_found.config(fg=font_color)

            self.strategy_snippet = (
                f"Set Strategy  default  confidence={self.image_horizon_instance.confidence}"
            )
            self.view.set_strategy_snippet.set(self.strategy_snippet)
            self.view.btn_copy_strategy_snippet["state"] = "normal"
            self.view.processing_done = True
        except Exception as exc:
            LOGGER.error(exc)
            self.view.label_statusBar.config(fg='RED')
            self.view.hint_msg.set(str(exc))
            self.view.processing_done = False

    def on_click_run_edge_detec_strategy(self):
        """Execute recognition using the edge detection strategy."""
        try:
            self.image_horizon_instance.set_strategy('edge')
            self.image_horizon_instance.edge_low_threshold = float(
                self.view.scale_low_thres_edge.get()
            )
            self.image_horizon_instance.edge_high_threshold = float(
                self.view.scale_high_thres_edge.get()
            )
            if (
                self.image_horizon_instance.edge_high_threshold
                < self.image_horizon_instance.edge_low_threshold
            ):
                self.reset_results()
                self.reset_images()
                self.view._threshold_error()
                self.view.processing_done = False
                return

            self.image_horizon_instance.confidence = float(
                self.view.scale_conf_lvl_edge.get()
            )
            self.image_horizon_instance.edge_sigma = float(
                self.view.scale_sigma_edge.get()
            )
            self.image_container._haystack_image_orig_size = self._take_screenshot()

            matcher = Cv2(self.image_container, self.image_horizon_instance)
            self.coord = matcher.find_num_of_matches()
            matcher.highlight_matches()

            self.haystack_image = self.image_container.get_haystack_image(
                format=ImageFormat.IMAGETK
            )
            self.view.canvas_desktop_img.itemconfig(
                self.view.desktop_img, image=self.haystack_image
            )

            num_of_matches_found = len(self.coord)
            max_peak = round(np.amax(self.image_horizon_instance.peakmap), 2)
            if max_peak < 0.75:
                result_msg = f"{num_of_matches_found} / max peak value below 0.75"
            else:
                result_msg = f"{num_of_matches_found} / {max_peak}"

            self.view.matches_found.set(result_msg)
            font_color = self.model.change_color_of_label(num_of_matches_found)
            self.view.label_matches_found.config(fg=font_color)
            self.view.btn_edge_detec_debugger["state"] = "normal"

            self.strategy_snippet = (
                f"Set Strategy  edge  edge_sigma={self.image_horizon_instance.edge_sigma}  "
                f"edge_low_threshold={self.image_horizon_instance.edge_low_threshold}  "
                f"edge_high_threshold={self.image_horizon_instance.edge_high_threshold}  "
                f"edge_preprocess={self.image_horizon_instance.edge_preprocess}  "
                f"edge_kernel_size={self.image_horizon_instance.edge_kernel_size}  "
                f"validate_match={self.image_horizon_instance.validate_match}  "
                f"validation_margin={self.image_horizon_instance.validation_margin}  "
                f"confidence={self.image_horizon_instance.confidence}"
            )
            self.view.set_strategy_snippet.set(self.strategy_snippet)
            self.view.btn_copy_strategy_snippet["state"] = "normal"
            self.view.processing_done = True
        except Exception as exc:
            LOGGER.error(exc)
            self.view.label_statusBar.config(fg='RED')
            self.view.hint_msg.set(str(exc))
            self.view.processing_done = False

    def on_click_plot_results_edge(self):
        """Display detailed edge detection results in a matplotlib window."""
        title = (
            f"{self.view.matches_found.get()} matches (confidence: "
            f"{self.image_horizon_instance.confidence})"
        )
        try:
            self.model.plot_result(
                self.image_container.get_needle_image(ImageFormat.NUMPYARRAY),
                self.image_container.get_haystack_image_orig_size(
                    ImageFormat.NUMPYARRAY
                ),
                self.image_horizon_instance.needle_edge,
                self.image_horizon_instance.haystack_edge,
                self.image_horizon_instance.peakmap,
                title,
                self.coord,
            )
        except ImportError as exc:
            # Provide feedback if the optional debug dependencies are missing
            self.view.hint_msg.set(str(exc))
