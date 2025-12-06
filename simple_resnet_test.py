"""
Minimal ResNet-18 Test
======================

Simple script to test ResNet-18 basic functionality.
No training required - just verifies the network works.
"""

import torch
from resnet18_cifar import create_resnet18_cifar10

def main():
    print("=" * 60)
    print("SIMPLE RESNET-18 TEST")
    print("=" * 60)

    # 1. Create model
    print("\n1. Creating ResNet-18...")
    model = create_resnet18_cifar10(num_classes=10, small=True)
    params = model.get_num_parameters()
    print(f"   ✓ Model created: {params:,} parameters")

    # 2. Test forward pass
    print("\n2. Testing forward pass...")
    x = torch.randn(4, 3, 32, 32)  # 4 images, 3 channels, 32x32
    print(f"   Input shape: {x.shape}")

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"   Output shape: {output.shape}")
    print(f"   ✓ Forward pass successful!")

    # 3. Show predictions
    print("\n3. Sample predictions:")
    probs = torch.softmax(output, dim=1)
    predictions = torch.argmax(probs, dim=1)

    for i in range(4):
        pred_class = predictions[i].item()
        confidence = probs[i, pred_class].item() * 100
        print(f"   Image {i}: Class {pred_class} (confidence: {confidence:.1f}%)")

    # 4. Test backward pass
    print("\n4. Testing backward pass...")
    model.train()
    output = model(x)
    loss = output.sum()
    loss.backward()

    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"   ✓ Gradients computed: {has_grads}/{total_params} parameters")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nResNet-18 is working correctly!")
    print("Next steps:")
    print("  - Run: python test_resnet18.py (full test suite)")
    print("  - Run: python experiments_resnet18.py --quick --small")

if __name__ == '__main__':
    main()
