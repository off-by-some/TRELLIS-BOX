# Memory Leak Fixes

This document summarizes all the memory leak fixes implemented in the TRELLIS 3D application.

## Issues Identified and Fixed

### 1. Global Refiner Instance Leak ✅
**Problem:** The `refiner` variable in `app.py` was a global variable that was never properly cleaned up, keeping the Stable Diffusion XL model in memory even when not in use.

**Fix:**
- Moved refiner from global variable to `st.session_state.refiner`
- Updated all references to use session state
- Added proper cleanup calls to `unload()` method after use
- Added `cleanup()` and `__del__()` methods to ImageRefiner class for proper resource cleanup

**Files Modified:**
- `app.py`: Lines 82-119, 140-142, 190-192
- `trellis/pipelines/image_refiner.py`: Lines 120-142

---

### 2. Rembg Session Not Being Closed ✅
**Problem:** The `rembg_session` was created for background removal but never explicitly closed or cleaned up, causing accumulation of background removal model resources.

**Fix:**
- Changed `getattr()` check to proper `hasattr()` check
- Added `cleanup()` method to `TrellisImageTo3DPipeline` class
- The cleanup method properly deletes the rembg_session and releases resources
- Called periodically during session state cleanup

**Files Modified:**
- `trellis/pipelines/trellis_image_to_3d.py`: Lines 114, 482-499

---

### 3. Session State Accumulation ✅
**Problem:** Large objects (images, videos, 3D models, tensors) were stored in Streamlit session state and never cleaned up, causing memory to grow over time.

**Fix:**
- Created comprehensive `cleanup_session_state()` function
- Clears specific large objects: `generated_video`, `generated_glb`, `generated_state`, `uploaded_image`, `processed_preview`, `processed_image`
- Resets image preview render counts to prevent integer overflow
- Calls pipeline and refiner cleanup methods
- Integrated cleanup into all clear button actions
- Calls `reduce_memory_usage()` for GPU memory management

**Files Modified:**
- `app.py`: Lines 51-97, 598-599, 613, 709-710, 730-731, 738-739, 773, 889-890, 900-901

---

### 4. Image Hashing Creating Unnecessary Copies ✅
**Problem:** The `_get_image_hash()` function in `image_preview.py` was creating full byte copies of images just to detect changes, consuming significant memory.

**Fix:**
- Replaced full image serialization with object ID + metadata approach
- Only samples a small 10x10 pixel region from center for verification
- Reduces memory usage from O(image_size) to O(1)
- Maintains change detection accuracy while being much faster

**Files Modified:**
- `webui/image_preview.py`: Lines 16-45

---

### 5. Periodic Cleanup ✅
**Problem:** Even with manual cleanup, resources could accumulate over multiple interactions without any periodic maintenance.

**Fix:**
- Added `periodic_cleanup()` function that runs on every page interaction
- Tracks interaction count with `cleanup_counter`
- Performs lightweight cleanup every 5 interactions
- Performs aggressive cleanup (including temp file removal) every 20 interactions
- Automatically called at the start of `main()` function

**Files Modified:**
- `app.py`: Lines 100-125, 577-578

---

### 6. Processed Preview Image Accumulation ✅
**Problem:** The `processed_preview` images were regenerated and stored without cleaning up old references, causing accumulation.

**Fix:**
- Added explicit deletion of old preview before storing new one
- Ensures only one processed preview exists at a time
- Reduces peak memory usage during image preprocessing

**Files Modified:**
- `app.py`: Lines 662-666

---

## Additional Cleanup Methods Added

### TrellisImageTo3DPipeline.cleanup()
Cleans up pipeline resources including:
- Deletes rembg_session
- Clears CUDA cache
- Synchronizes CUDA operations

### ImageRefiner.cleanup()
Fully releases refiner resources:
- Moves model to CPU
- Deletes the pipeline
- Clears CUDA cache
- Added `__del__()` destructor for automatic cleanup

---

## Best Practices Implemented

1. **Explicit Resource Management**: All large objects are explicitly deleted after use
2. **Session State Hygiene**: Regular cleanup of accumulated state
3. **GPU Memory Management**: Frequent `torch.cuda.empty_cache()` calls
4. **Temp File Cleanup**: Periodic removal of old temporary files
5. **Garbage Collection**: Forced GC calls after large operations
6. **Memory-Efficient Operations**: Reduced memory footprint of common operations

---

## Testing Recommendations

To verify these fixes are working:

1. **Monitor GPU Memory**: Use `nvidia-smi -l 1` to watch GPU memory usage over time
2. **Run Multiple Generations**: Generate 10-20 models in a row and verify memory doesn't grow
3. **Clear Actions**: Test all clear buttons and verify memory is released
4. **Long Running Session**: Leave the app running and perform various operations over hours
5. **Check Temp Directory**: Verify old files are being cleaned up

---

## Expected Memory Behavior

After these fixes:
- Memory usage should stabilize after initial model loading
- Each generation should return to baseline memory after completion
- Clear actions should immediately release memory
- No gradual memory increase over multiple operations
- Temp directory size should remain bounded

---

## Configuration

Memory management can be tuned via:
- `cleanup_counter` intervals (currently 5 and 20 interactions)
- `cleanup_temp_files()` age threshold (currently 1 hour)
- Image preview render count reset threshold (currently 1000)

---

Generated: $(date)

