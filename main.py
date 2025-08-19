from menu import run_menu


if __name__ == "__main__":
    try:
        run_menu()
    except RuntimeError as e:
        print(f"Fehler: {e}")
        print("Hinweis: Installiere opencv-contrib-python und stelle sicher, dass NumPy/Torch kompatibel sind.")

