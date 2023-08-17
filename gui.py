import threading
import customtkinter as ctk
import time

#Define colors
colors = {
        "background": "#3d392d",
        "text_input_bg_color": "#1a2014",
        "text_output_fg_color": "#181612",
        "button_bg_color": "#344f58",
        "button_fg_color": "#447c89"
        }

#Define fonts
fonts = {
        "default": ("Courier New", 16),
        }

#Create the main window
window = ctk.CTk()
window.geometry("400x800")
window.title("AI Chatbot")
window.resizable(width=False, height=False)

frame = ctk.CTkFrame(master=window,
                                width=400,
                                height=800,
                                corner_radius=10,
                                fg_color = colors["background"])
frame.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

prompt_label = ctk.CTkLabel(window, 100, 15, text="User Prompt", font=fonts["default"], fg_color = colors["background"])
prompt_label.pack(pady=10)

input_textbox = ctk.CTkEntry(window, 340, 30, font=fonts["default"], corner_radius=5)
input_textbox.pack()

load_button = ctk.CTkButton(window, 100, 30, text="Load Prompt", font=fonts["default"], corner_radius=5, fg_color=colors["button_fg_color"])                    
load_button.pack(pady=30    )

output_label = ctk.CTkLabel(window, 100, 15, text="Model Output", font=fonts["default"], fg_color = colors["background"])                                
output_label.pack(pady=10)

model_output_textbox = ctk.CTkLabel(window, 340, 575, text="", font=fonts["default"], corner_radius=5, fg_color=colors["text_output_fg_color"], anchor="nw")                                        
model_output_textbox.pack() 

def window_slaves():
    time.sleep(2)
    print(window.slaves())


def main():
    task_slaves = threading.Thread(target=window_slaves)
    task_slaves.start()

    # Start the main loop
    window.mainloop()


    task_slaves.join()

 
    

if __name__ == "__main__":
    main()