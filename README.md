# Transcript

כלי שורת־פקודה שמקבל קובץ וידאו בפורמט MP4, מפיק תמלול בעזרת מודל Whisper, ושומר צילומי מסך בכל פעם שמתגלה שינוי ויזואלי משמעותי.

## תלותים

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) צריך להיות זמין במשתנה הסביבה `PATH` כדי ש-Whisper תוכל לחלץ אודיו.
- חבילות פייתון: `openai-whisper`, `torch`, `opencv-python`, `numpy` (ראו דוגמת התקנה למטה).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install openai-whisper torch opencv-python numpy
```

## שימוש

```powershell
python transcribe_video.py path\to\video.mp4 output_folder `
	--model base `
	--language he `
	--diff-threshold 12 `
	--min-interval 2.5 `
	--transcript-format json
```

- `output_folder\transcript.json` או `transcript.srt` יכיל את התמלול.
- צילומי המסך יישמרו תחת `output_folder\frames`. בכל שם קובץ יופיע מיקום בזמן (שעות-דקות-שניות-אלפיות).
- `output_folder\frames\metadata.json` מסכם את המקומות בזמן והקבצים שנוצרו.

### פרמטרים שימושיים

- `--model` מאפשר לבחור מודל Whisper אחר (למשל `small` לזיהוי מדויק יותר אך איטי).
- `--language` אפשר להשאיר ריק כדי לתת למודל לזהות שפה אוטומטית.
- `--diff-threshold` קובע כמה שינוי פיקסלים נדרש כדי לשמור פריים (ערך גבוה יותר = פחות צילומים).
- `--min-interval` מונע שמירה של פריימים קרובים מדי בזמן.
- `--resize-width` מאפשר לשמור תמונות מוקטנות (למשל `--resize-width 1280`).

## תוצאות

בסיום הריצה הסקריפט ידפיס לאן נשמרו התמלול וצילומי המסך וכמה צילומי מפתח זוהו.