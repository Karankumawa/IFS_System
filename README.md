# Indian Festival Recognition

One desktop application recognizes nine Indian festivals from a live camera or
an image file and plays a local song for the detected festival.

## Run the app

The current `venv` does not yet contain the required packages. Run these
commands in PowerShell from this project folder:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe test_load.py
.\venv\Scripts\python.exe app.py
```

`test_load.py` must print `SUCCESS` before opening the app.

## Use the desktop app

1. Click **Start Camera** for live preview and live festival detection.
2. Click **Capture Current Frame** to analyze the current camera image once.
3. Or click **Choose Image** to analyze an image from your computer.
4. Select a festival in the audio panel and click **Choose Song File** to add
   a local `.mp3`, `.wav`, or `.ogg` track for it.

Live detection waits for three consecutive matching predictions before changing
the detected festival or song. Click **Stop Song** at any time.

The project includes `Happy_Diwali.mp3`, so DIWALI can play immediately. Add
your own local tracks for the other festivals using the app. Your selections
are saved in `audio_mapping.json` on this computer and are not committed to Git.

## Project files

- `app.py` — the single desktop user interface.
- `festival_model.keras` and `festival_model.labels.json` — the supplied model
  and its required label order.
- `festival_config.py` — shared model, dataset, and audio settings.
- `train_model.py` — optional model retraining script.
- `test_load.py` — model smoke test.
- `dataset/festivals_classification` — classifier data.

The training data is imbalanced and its folder named `test` is currently used
for validation. Create a separate unseen test split before reporting final
accuracy from a retrained model.
