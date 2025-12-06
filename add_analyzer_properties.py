"""
Example: Adding Properties to ModuliSpaceAnalyzer
==================================================

This shows how to add the missing properties that test_resnet18.py expects.

Add these properties to the ModuliSpaceAnalyzer class in moduli_space_pruning.py
after the __init__ method (around line 179).
"""

# ADD THESE PROPERTIES TO ModuliSpaceAnalyzer CLASS:

class ModuliSpaceAnalyzerWithProperties:
    """Example showing the properties to add"""

    # ... existing __init__ and methods ...

    @property
    def total_coordinates(self):
        """Get total number of coordinates (edges in quiver)"""
        return len(self.quiver.edges)

    @property
    def moduli_dimension(self):
        """
        Get moduli space dimension.
        Formula: dim(M) = #E - #Ṽ (edges minus hidden vertices)
        """
        num_edges = len(self.quiver.edges)
        num_vertices = len(self.quiver.vertices)
        return num_edges - num_vertices

    @property
    def coordinates(self):
        """
        Get list of analyzed coordinate objects.
        Returns empty list if analyze_dataset hasn't been called yet.
        """
        return list(self.coordinate_stats.values())

    @property
    def num_analyzed_coordinates(self):
        """Get number of coordinates that have been analyzed"""
        return len(self.coordinate_stats)


# USAGE EXAMPLE:
if __name__ == '__main__':
    """
    After adding the properties above to moduli_space_pruning.py,
    you can use them like this:
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from resnet18_cifar import create_resnet18_cifar10
    from moduli_space_pruning import ModuliSpaceAnalyzer

    # Create model
    model = create_resnet18_cifar10(num_classes=10, small=True)

    # Create analyzer
    analyzer = ModuliSpaceAnalyzer(model)

    # Access properties BEFORE analysis
    print(f"Total coordinates: {analyzer.total_coordinates:,}")
    print(f"Moduli dimension: {analyzer.moduli_dimension:,}")
    print(f"Analyzed coordinates: {analyzer.num_analyzed_coordinates}")
    print(f"Coordinate objects: {len(analyzer.coordinates)}")

    # Create dummy data
    X = torch.randn(50, 3, 32, 32)
    y = torch.randint(0, 10, (50,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)

    # Analyze
    analyzer.analyze_dataset(dataloader, max_samples=50)

    # Access properties AFTER analysis
    print(f"\nAfter analysis:")
    print(f"Total coordinates: {analyzer.total_coordinates:,}")
    print(f"Moduli dimension: {analyzer.moduli_dimension:,}")
    print(f"Analyzed coordinates: {analyzer.num_analyzed_coordinates:,}")
    print(f"Coordinate objects: {len(analyzer.coordinates)}")

    # Access individual coordinate statistics
    for coord in analyzer.coordinates[:5]:
        print(f"\nCoordinate {coord.edge_idx}:")
        print(f"  Usage rate: {coord.usage_rate:.4f}")
        print(f"  Variance: {coord.variance:.6f}")
        print(f"  Importance: {coord.importance_score:.6f}")
