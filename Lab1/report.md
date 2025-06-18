# Simple Report – Multimodal Classifier with Google Teachable Machine

### Goal  
Build a small “retail helper” that can tell three objects apart by **looking** at them (webcam) and can tell two music artists apart by **listening** (microphone).

---

## 1. What We Trained

| Part | Classes & Sample Count | How We Collected Data |
|------|-----------------------|-----------------------|
| **Image Model** | *Smartphones, Laptops, Bottles* (≈30 photos each) | Used webcam; showed each item from different angles. |
| **Audio Model** | *Queen, Hozier* (≈20 short clips each) | Played songs on laptop speaker; room had normal background noise. |
| **Combined Model** | Teachable Machine’s “Combine” block joins the two models | No extra data needed. |

---

## 2. How Training Went  

* **Image accuracy:** about **96 %**  
* **Audio accuracy:** about **93 %**  
* Training took only a couple of minutes in the browser.

---

## 3. Test Results

### 3.1  Image Model Confusion Matrix  

*(Blue squares = correct guesses)*  

![Confusion Matrix](confusion.png)

*Out of 52 tests, 50 were right → ~96 %.*

### 3.2  Quick Audio Check  

* Queen songs were usually tagged **Queen**.  
* Hozier songs were usually tagged **Hozier**.  
* During silent pauses the prediction sometimes flipped back and forth.

---

## 4. What Went Wrong (and Why)

| Mis-step | Likely Reason | Easy Fix |
|----------|---------------|----------|
| Holding the laptop **upright** looked like a phone. | Phones are tall and thin → model saw similar shape. | Add more photos of the laptop held upright. |
| One laptop photo showed as **Bottle**. | Screen glare looked like a shiny bottle. | Take photos in softer light. |
| Audio model jumps between artists when there’s silence. | With no sound, the model isn’t sure. | Ignore frames with very low volume or average the result over 1–2 seconds. |

---

## 5. Live Demo Highlights  

1. Show bottle → says **Bottle** (fast).  
2. Show laptop sideways → **Laptop** (correct).  
3. Show laptop upright → **Phone** (mistake).  
4. Play *Bohemian Rhapsody* → **Queen** almost all the time.  
5. Short silent gap in Hozier song → guesses flicker.

*A short demo video (`demo_multimodal.mp4`) is included in my submission folder.*

---

## 6. Take-aways & Next Steps  

* Even with a tiny dataset, Teachable Machine gives good accuracy.  
* **More varied photos** (different angles, lighting) will cut down the image mistakes.  
* **Smoothing audio output** (ignore silence) will steady the artist guesses.  
* Future idea: export the trained model to a phone so shoppers can use it on-device.

---
