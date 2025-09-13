#Python Include Test
try:
    import torch
    print("PyTorch is successfully imported!")
    print(f"PyTorch version: {torch.__version__}")

    #Check Intel GPU is available
    if torch.xpu.is_available():
        print(f"Intel GPU is available! Device count: {torch.xpu.device_count()}")
        print(f"Current GPU Device: {torch.xpu.get_device_name(0)}")
    else:
        print("Intel GPU is not available. Using CPU.")

    #Calculate simple tensor by GPU
    test_tensor = torch.tensor([1, 2, 3, 4, 5]).to("xpu")
    print(f"Tst tensor created: {test_tensor}")
    print(f"Tensor shape: {test_tensor.shape}")
    print(f"Tensor device: {test_tensor.device}")

    squared_tensor = test_tensor**2
    print(f"Basic operation test (squaring): {squared_tensor}")

    print("\n PyTorch is working by Intel GPU correctly!")

except ImportError as e:
    print("PyTorch is not installed or not available")
    print(f"Error details: {e}")
    print("Please install PyTorch using: pip install torch")

except Exception as e:
    print(f"An error occurred while testing PyTorch: {e}")
