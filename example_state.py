from webui.state_manager import StateManager

# Should be able to set the state transparently
StateManager.uploaded_image = "test"
StateManager.processed_preview = "test"
StateManager.generated_video = "test"
StateManager.generated_glb = "test"
StateManager.generated_state = "test"



with StateManager.uploaded_image:
    print("running uploaded_image", StateManager.uploaded_image)

with StateManager.processed_preview:
    print("running processed_preview", StateManager.processed_preview)

with StateManager.generated_video:
    print("running generated_video", StateManager.generated_video)



# Test that the state is updated
StateManager.uploaded_image = "test2"
StateManager.processed_preview = "test2"
StateManager.generated_video = "test2"
StateManager.generated_glb = "test2"
StateManager.generated_state = "test2"

# Should see the print statements again