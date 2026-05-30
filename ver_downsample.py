import os
from tkinter import Button, Label, Tk, filedialog, messagebox


# Funktion som körs när man klickar på "Välj fil och kör"
def välj_och_kör():
    # Öppna dialogrutan för att välja fil
    input_filepath = filedialog.askopenfilename(
        title="Select a datafile to downsample",
        filetypes=[("Textfiler", "*.txt"), ("Alla filer", "*.*")],
    )

    # Om användaren valde en fil, kör nedsamplingen
    if input_filepath:
        try:
            file_dir, file_name = os.path.split(input_filepath)
            name, ext = os.path.splitext(file_name)
            output_filepath = os.path.join(file_dir, f"{name}_250_Hz{ext}")

            with open(input_filepath, "r") as infile, open(
                output_filepath, "w"
            ) as outfile:
                for index, line in enumerate(infile):
                    if index % 4 == 0:
                        outfile.write(line)

            # Visa en rutan som berättar att det är klart
            messagebox.showinfo(
                "Done!",
                f"The file is downsampled and saved as:\n{output_filepath}",
            )

        except Exception as e:
            messagebox.showerror("Fel", f"Ett fel uppstod: {e}")
    else:
        # Om användaren klickade avbryt i fildialogen, gör ingenting (återgå till fönstret)
        pass


# --- Skapa huvudfönstret ---
root = Tk()
root.title("ver-downsampling")
root.geometry("400x200")  # Sätter storleken på fönstret (bredd x höjd)

# 1. Förklaringstext (Här kan du skriva vad du vill)
förklaring = (
    "This program downsamples Labchart exported .TXT files from 1000 Hz to 250 Hz.\n\n"
    "Click on the button to select the file to downsample. "
    "The file will be saved with the same name plus _250_Hz, "
    
)

lbl_text = Label(
    root, text=förklaring, justify="left", wraplength=360, padx=20, pady=20
)
lbl_text.pack()

# 2. Knapp för att välja fil och köra programmet
btn_run = Button(
    root, text="Select file to downsample", command=välj_och_kör, bg="white", fg="black"
)
btn_run.pack(pady=5)

# 3. Exit-knapp för att stänga fönstret
btn_exit = Button(root, text="Exit", command=root.destroy, bg="white", fg="black")
btn_exit.pack(pady=5)

# Starta fönster-loopen
root.mainloop()