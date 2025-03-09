import customtkinter as ctk
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk 

plt.style.use('dark_background')

# Dummy function to simulate the forecasting of oil futures
def get_oil_futures_data():
    # This would normally be replaced with your forecasting data or API call
    data = {
        'Date': pd.date_range(start='2025-03-01', periods=10, freq='D'),
        'Forecasted Price': [75 + i * 0.5 for i in range(10)]  # Simulating some increasing prices
    }
    return pd.DataFrame(data)

class OilFuturesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CMEnergy Portal")
        
        # Set the size of the window
        self.set_window_size(50, 50)
        root.after(0, lambda:root.state('zoomed'))
        
        # Configure CTk style
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root.iconbitmap(default="project/media/cme.ico")

        self.load_logo_image()
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Title label
        self.title_label = ctk.CTkLabel(self.main_frame, text="Oil Futures Forecasting", font=("Arial", 24))
        self.title_label.pack(pady=20)
        
        # Forecast data button
        self.load_button = ctk.CTkButton(self.main_frame, text="Load Forecasting Data", command=self.load_forecasting_data)
        self.load_button.pack(pady=10)
        
        # Create a frame for the graph
        self.graph_frame = ctk.CTkFrame(self.main_frame)
        self.graph_frame.pack(pady=20, fill="both", expand=True)
        
        # Create a table display frame
        self.table_frame = ctk.CTkFrame(self.main_frame)
        self.table_frame.pack(pady=20, fill="both", expand=True)

    def load_logo_image(self):
        # Load an image (make sure to replace the path with your own image path)
        logo_path = "project/media/cmelong.png"  # Replace this with the actual path to your logo
        logo_image = Image.open(logo_path)  # Open the image file
        logo_image = logo_image.resize((200, 35))  # Resize image (optional)
        self.logo = ImageTk.PhotoImage(logo_image)  # Convert image to Tkinter-compatible format
        
        # Create a frame to hold both image and text
        self.logo_frame = ctk.CTkFrame(self.root)
        self.logo_frame.pack(side="bottom", anchor="w", pady=10, padx=15, fill="x")

        # Create label for the image
        self.logo_label = ctk.CTkLabel(self.logo_frame, image=self.logo, text="", width=200, height=35)  # Set the width and height of the image label
        self.logo_label.grid(row=0, column=0, padx=(0, 10))  # Place it at the first column

        # Create label for the text
        self.text_label = ctk.CTkLabel(self.logo_frame, text="CMEnergy", font=("Arial", 22, "bold"), text_color="black", anchor="e")
        self.text_label.grid(row=0, column=1, padx=15, sticky="e")  # Place it at the second column

        # Make sure the logo_frame expands to fill the window and the second column can take the available space
        self.logo_frame.grid_columnconfigure(0, weight=0)  # Keep the first column (logo) with no weight
        self.logo_frame.grid_columnconfigure(1, weight=1)  # Allow the second column (text) to take up all remaining space

        # Ensure the row can expand to fill available height
        self.logo_frame.grid_rowconfigure(0, weight=1)

    def set_window_size(self, width_percentage, height_percentage):
        # Get the screen's width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate the width and height based on the given percentage
        window_width = int(screen_width * (width_percentage / 100))
        window_height = int(screen_height * (height_percentage / 100))
        
        # Set the window size using the calculated width and height
        self.root.geometry(f"{window_width}x{window_height}+0+0")

        
    def load_forecasting_data(self):
        # Get dummy oil futures data
        data = get_oil_futures_data()
        
        # Display data in a plot
        self.display_graph(data)
        
        # Display data in a table
        self.display_table(data)
        
    def display_graph(self, data):
        # Clear previous content
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(data['Date'], data['Forecasted Price'], marker='o', color='b', label='Forecasted Price')
        ax.set_title('Oil Futures Price Forecasting')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        
        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def display_table(self, data):
        # Clear previous table content
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        # Create a simple table to display data
        table = ctk.CTkLabel(self.table_frame, text=data.to_string(index=False), font=("Courier", 12))
        table.pack(padx=10, pady=10)

# Initialize the Tkinter root window
root = ctk.CTk()

# Create the app instance
app = OilFuturesApp(root)

# Run the app
root.mainloop()
