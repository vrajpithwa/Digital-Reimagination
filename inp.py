import torch
from diffusers import StableDiffusionInpaintPipeline
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
from datetime import datetime

class GenerativeInpaintingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Generative Inpainting Tool")
        
        # Initialize variables
        self.image = None
        self.mask = None
        self.display_image = None
        self.display_scale = 1.0
        self.drawing = False
        self.brush_size = 15
        self.last_x = None
        self.last_y = None
        self.original_image = None
        self.current_image_path = None
        
        # Initialize undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []
        
        # Load model
        print("Loading Stable Diffusion model...")
        model_id = "stabilityai/stable-diffusion-2-inpainting"
        
        try:
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
                self.pipeline.enable_xformers_memory_efficient_attention()
            else:
                self.pipeline = self.pipeline.to("cpu")
            
            # Enable memory optimizations
            self.pipeline.enable_attention_slicing()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.destroy()
            return
        
        self.setup_ui()
        
        # Create output directory if it doesn't exist
        self.output_dir = "inpainting_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def setup_ui(self):
        # Main container
        self.main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel (left side)
        self.control_panel = ttk.Frame(self.main_container)
        self.main_container.add(self.control_panel, weight=1)
        
        # Image area (right side)
        self.image_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.image_frame, weight=4)
        
        # Setup controls
        self.setup_control_panel()
        
        # Setup canvas
        self.setup_canvas()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready")
        
        # Add menu bar
        self.setup_menu()

    def setup_control_panel(self):
        # Load image button
        ttk.Button(
            self.control_panel,
            text="Load Image",
            command=self.load_image
        ).pack(fill=tk.X, padx=5, pady=5)
        
        # Brush size control
        brush_frame = ttk.LabelFrame(self.control_panel, text="Brush Size")
        brush_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.brush_size_var = tk.IntVar(value=self.brush_size)
        brush_scale = ttk.Scale(
            brush_frame,
            from_=1,
            to=50,
            variable=self.brush_size_var,
            command=self.update_brush_size
        )
        brush_scale.pack(fill=tk.X, padx=5, pady=5)
        
        # Prompt input
        prompt_frame = ttk.LabelFrame(self.control_panel, text="Prompt")
        prompt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.prompt_text = tk.Text(prompt_frame, height=3, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Negative prompt input
        neg_prompt_frame = ttk.LabelFrame(self.control_panel, text="Negative Prompt")
        neg_prompt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.negative_prompt_text = tk.Text(neg_prompt_frame, height=3, wrap=tk.WORD)
        self.negative_prompt_text.pack(fill=tk.X, padx=5, pady=5)
        self.negative_prompt_text.insert('1.0', "blur, ugly, poorly drawn, bad anatomy, deformed")
        
        # Generate button
        ttk.Button(
            self.control_panel,
            text="Generate",
            command=self.generate_inpainting
        ).pack(fill=tk.X, padx=5, pady=5)
        
        # Clear mask button
        ttk.Button(
            self.control_panel,
            text="Clear Mask",
            command=self.clear_mask
        ).pack(fill=tk.X, padx=5, pady=5)
        
        # Add save buttons
        self.save_frame = ttk.Frame(self.control_panel)
        self.save_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            self.save_frame,
            text="Save Result",
            command=self.save_result
        ).pack(side=tk.LEFT, expand=True, padx=2)
        
        ttk.Button(
            self.save_frame,
            text="Save As...",
            command=self.save_result_as
        ).pack(side=tk.LEFT, expand=True, padx=2)

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Save Result", command=self.save_result)
        file_menu.add_command(label="Save As...", command=self.save_result_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo_last_stroke)
        edit_menu.add_command(label="Redo", command=self.redo_last_stroke)
        edit_menu.add_command(label="Clear Mask", command=self.clear_mask)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show Original", command=self.show_original)
        view_menu.add_command(label="Show Mask", command=self.show_mask)

    def setup_canvas(self):
        # Create canvas frame with scrollbars
        self.canvas_frame = ttk.Frame(self.image_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollbars
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create canvas
        self.canvas = tk.Canvas(
            self.canvas_frame,
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set,
            bg='gray'
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure scrollbars
        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # Bind mouse wheel for zooming
        self.canvas.bind("<Control-MouseWheel>", self.zoom)  # Windows
        self.canvas.bind("<Control-Button-4>", self.zoom)    # Linux scroll up
        self.canvas.bind("<Control-Button-5>", self.zoom)    # Linux scroll down

        # Create image container on canvas
        self.container = self.canvas.create_image(0, 0, anchor="nw")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and convert image to RGBA
                self.image = Image.open(file_path).convert('RGBA')
                self.original_image = self.image.copy()
                self.current_image_path = file_path
                
                # Clear existing mask
                self.mask = None
                
                # Update display
                self.update_display()
                
                self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def update_brush_size(self, value):
        self.brush_size = int(float(value))

    def start_drawing(self, event):
        if self.image is None:
            return
            
        self.drawing = True
        self.last_x = self.canvas.canvasx(event.x)
        self.last_y = self.canvas.canvasy(event.y)
        
        # Save current mask state for undo
        if self.mask is not None:
            self.undo_stack.append(self.mask.copy())
        self.redo_stack.clear()

    def draw(self, event):
        if not self.drawing or self.image is None:
            return
            
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Draw on the mask
        if self.mask is None:
            self.mask = Image.new('L', self.image.size, 0)
        draw = ImageDraw.Draw(self.mask)
        
        # Draw line on mask
        draw.line(
            [self.last_x/self.display_scale, 
             self.last_y/self.display_scale, 
             x/self.display_scale, 
             y/self.display_scale],
            fill=255,
            width=int(self.brush_size/self.display_scale)
        )
        
        # Update display
        self.update_display()
        
        self.last_x = x
        self.last_y = y

    def stop_drawing(self, event):
        self.drawing = False

    def zoom(self, event):
        if not self.image:
            return
            
        # Get old scale
        old_scale = self.display_scale
        
        # Zoom in/out
        if event.delta > 0 or event.num == 4:
            self.display_scale *= 1.1
        elif event.delta < 0 or event.num == 5:
            self.display_scale /= 1.1
        
        # Limit zoom
        self.display_scale = min(max(0.1, self.display_scale), 5.0)
        
        # Update display if scale changed
        if old_scale != self.display_scale:
            self.update_display()

    def update_display(self):
        if self.image is None:
            return
            
        # Create a copy of the image
        display_image = self.image.copy()
        
        # If we have a mask, overlay it
        if self.mask is not None:
            # Create red overlay for mask
            overlay = Image.new('RGBA', self.image.size, (255, 0, 0, 0))
            mask_rgba = Image.new('RGBA', self.image.size, (255, 0, 0, 128))
            overlay.paste(mask_rgba, mask=self.mask)
            
            # Composite the overlay onto the image
            display_image = Image.alpha_composite(
                display_image.convert('RGBA'), 
                overlay
            )
        
        # Resize image according to scale
        display_size = tuple(
            int(dim * self.display_scale) for dim in display_image.size
        )
        display_image = display_image.resize(
            display_size, 
            Image.Resampling.LANCZOS
        )
        
        # Update canvas
        self.display_image = ImageTk.PhotoImage(display_image)
        self.canvas.itemconfig(self.container, image=self.display_image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def generate_inpainting(self):
        if self.image is None or self.mask is None:
            messagebox.showerror("Error", "Please load an image and draw a mask first")
            return
            
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showerror("Error", "Please enter a prompt")
            return
            
        negative_prompt = self.negative_prompt_text.get("1.0", tk.END).strip()
        
        # Continuation of generate_inpainting method
        try:
                # Prepare image and mask
                init_image = self.image.convert("RGB")
                mask_image = self.mask.convert("RGB")
                
                # Generate image
                self.status_var.set("Generating...")
                self.root.update()
                
                # Run inference
                image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    mask_image=mask_image,
                    num_inference_steps=50,
                    guidance_scale=7.5
                ).images[0]
                
                # Update image and display
                self.image = image.convert("RGBA")
                self.mask = None
                self.update_display()
                
                self.status_var.set("Generation complete!")
        except Exception as e:
                messagebox.showerror("Error", f"Generation failed: {str(e)}")
                self.status_var.set("Generation failed!")
            
    def clear_mask(self):
        if self.mask is not None:
            # Save current mask state for undo
            self.undo_stack.append(self.mask.copy())
            self.redo_stack.clear()
            
            # Clear mask
            self.mask = None
            self.update_display()
            self.status_var.set("Mask cleared")
    
    def undo_last_stroke(self):
        if self.undo_stack:
            # Save current state for redo
            if self.mask is not None:
                self.redo_stack.append(self.mask.copy())
            
            # Restore previous state
            self.mask = self.undo_stack.pop()
            self.update_display()
            self.status_var.set("Undo last stroke")
    
    def redo_last_stroke(self):
        if self.redo_stack:
            # Save current state for undo
            if self.mask is not None:
                self.undo_stack.append(self.mask.copy())
            
            # Restore next state
            self.mask = self.redo_stack.pop()
            self.update_display()
            self.status_var.set("Redo last stroke")
    
    def show_original(self):
        if self.original_image is not None:
            temp_image = self.image
            temp_mask = self.mask
            self.image = self.original_image.copy()
            self.mask = None
            self.update_display()
            self.root.after(1000, lambda: self.restore_state(temp_image, temp_mask))
    
    def restore_state(self, image, mask):
        self.image = image
        self.mask = mask
        self.update_display()
    
    def show_mask(self):
        if self.mask is not None:
            # Convert mask to visible image
            mask_display = Image.new('RGBA', self.image.size, (0, 0, 0, 0))
            mask_overlay = Image.new('RGBA', self.image.size, (255, 0, 0, 128))
            mask_display.paste(mask_overlay, mask=self.mask)
            
            # Temporarily show mask
            temp_image = self.image
            self.image = mask_display
            self.update_display()
            self.root.after(1000, lambda: self.restore_image(temp_image))
    
    def restore_image(self, image):
        self.image = image
        self.update_display()
    
    def save_result(self):
        if self.image is None:
            messagebox.showerror("Error", "No image to save")
            return
            
        if self.current_image_path:
            # Generate default filename based on original
            directory = os.path.dirname(self.current_image_path)
            filename = os.path.basename(self.current_image_path)
            base, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{base}_inpainted_{timestamp}{ext}"
            save_path = os.path.join(self.output_dir, new_filename)
            
            try:
                self.image.save(save_path)
                self.status_var.set(f"Saved result as: {new_filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
        else:
            self.save_result_as()
    
    def save_result_as(self):
        if self.image is None:
            messagebox.showerror("Error", "No image to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ],
            initialdir=self.output_dir
        )
        
        if file_path:
            try:
                self.image.save(file_path)
                self.status_var.set(f"Saved result as: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")

# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = GenerativeInpaintingApp(root)
    root.mainloop()