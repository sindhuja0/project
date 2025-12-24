
# README – Multi-Signal Stress Detector


## Logic: Stress Index Formulation

The goal of this project is to estimate a user’s stress level on a 0–100 scale by combining multiple interpretable physiological and behavioral signals extracted from a video stream.
Instead of relying on black-box emotion or stress APIs, the system explicitly computes raw signals and converts them into a final Stress_Index using transparent, rule-based logic.

### Signals Used

1. **Physiological Signal – Heart Rate (rPPG)**

* Extracted from subtle color changes in the facial skin, primarily from the green channel.

* Heart rate increases under stress due to activation of the sympathetic nervous system.

2. **Behavioral Signal – Eye Blink Activity**

* Derived from facial landmarks using the Eye Aspect Ratio (EAR).

* Increased or irregular blinking is commonly associated with cognitive load and stress.

###  Stress Index Formula
The Stress Index is computed by combining physiological arousal (heart rate) and behavioral load (blink rate) into a single normalized score in the range 0–100.

Heart Rate Normalization
```
hr_score = clamp ( (HR_current − HR_baseline) / HR_range )
```
Where:

* HR_current is the estimated heart rate from rPPG
* HR_baseline is the user’s personalized resting heart rate
* HR_range defines the HR increase corresponding to maximum stress

Blink Rate Normalization
```
blink_score = clamp ( (BlinkRate − Blink_baseline) / Blink_range )
```

Where:

* BlinkRate is blinks per minute
* Blink_baseline is the user’s baseline blink rate

Signal Fusion
```
stress_raw =  clamp (0.75 × hr_score + 0.25 × blink_score)

```

### Why This Logic?

* Personalized: Stress is measured relative to each user’s baseline, not a fixed population threshold.

* Robust: Reduces false positives caused by naturally high or low resting heart rates.

* Interpretable: Each component can be individually inspected and visualized.

* Consistent: Stress scores increase during stress tasks and remain low during relaxed states.

To improve temporal stability, the Stress Index is further smoothed over time, producing a realistic and user-friendly output.

## Model Choice: Pre-Trained Components Used

This system uses pre-trained models only for low-level perception, not for end-to-end stress prediction.

### 1. MediaPipe Face Mesh (Pre-trained)

**Why chosen:**

* Provides **468 accurate facial landmarks** in real time.
* Runs efficiently on CPU.
* Industry-standard and well-documented.
* MediaPipe Face Mesh is used strictly as a geometric feature extractor, not for emotion or stress classification.

### 2. Classical rPPG Signal Processing (No Black Box)

Instead of using an end-to-end deep learning model, a classical signal-processing pipeline is implemented.

Processing steps:

1. Extract mean green-channel intensity from the facial ROI.
2. Apply detrending and normalization.
3. Apply a band-pass filter to isolate heart-rate frequencies.
4. Estimate heart rate using FFT peak detection.

**Why this approach:**

1. Fully transparent and explainable.
2. Easier to debug and validate against ground truth signals.
3. Directly aligns with the task requirement to expose raw physiological signals.

## Trade-offs and Latency Handling

### Real-Time Constraints

* The system is designed to run locally on a standard laptop CPU.
* All processing is performed in a single real-time loop for simplicity and stability.

### Latency vs Stability Trade-off

* Heart rate estimation requires a temporal window (approximately 5–10 seconds).

* This introduces intentional latency, but significantly improves accuracy and reduces noise.

**Design decision:**

Signal stability and interpretability were prioritized over instantaneous, frame-level prediction, which is more appropriate for stress monitoring tasks.

###  Smoothing Strategy

* Exponential moving averages are applied to:

    * Heart rate
    * Stress Index

This results in:

* Reduced false spikes
* Smoother visualizations
* More reliable stress trends over time



