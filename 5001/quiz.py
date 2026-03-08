import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_custom_shapes():
    """
    Draws 3 separate 3D shapes similar to the output of Code 1,
    while maintaining cleaner and more modular code.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Function to add a cuboid to the plot
    def add_cuboid(ax, vertices, faces, colors):
        for face, color in zip(faces, colors):
            poly3d = Poly3DCollection([face], facecolors=color, linewidths=1, alpha=1)
            ax.add_collection3d(poly3d)

    # Shape 1: Cuboid
    vertices1 = [
        [0, 0, 0], [1, 0, 0], [1, 5, 0], [0, 5, 0],
        [0, 0, 1], [1, 0, 1], [1, 5, 1], [0, 5, 1]
    ]
    faces1 = [
        [vertices1[4], vertices1[5], vertices1[6], vertices1[7]],  # Top face
        [vertices1[0], vertices1[1], vertices1[5], vertices1[4]],  # Side face 1
        [vertices1[0], vertices1[3], vertices1[7], vertices1[4]]   # Side face 2
    ]
    colors1 = ['magenta', 'yellow', 'cyan']

    # Shape 2: Another Cuboid
    vertices2 = [
        [0, 4, 0], [5, 4, 0], [5, 5, 0], [0, 5, 0],
        [0, 4, 1], [5, 4, 1], [5, 5, 1], [0, 5, 1]
    ]
    faces2 = [
        [vertices2[4], vertices2[5], vertices2[6], vertices2[7]],  # Top face
        [vertices2[0], vertices2[1], vertices2[5], vertices2[4]]   # Side face
    ]
    colors2 = ['magenta', 'yellow']

    # Shape 3: Slanted Top Cuboid
    a, b = 5.45, 4.05
    vertices3 = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, a], [1, 0, a], [1, 1, b], [0, 1, b]
    ]
    faces3 = [
        [vertices3[4], vertices3[5], vertices3[6], vertices3[7]],  # Top face
        [vertices3[0], vertices3[1], vertices3[5], vertices3[4]],  # Side face 1
        [vertices3[0], vertices3[3], vertices3[7], vertices3[4]]   # Side face 2
    ]
    colors3 = ['magenta', 'yellow', 'cyan']

    # Add shapes to the plot
    add_cuboid(ax, vertices1, faces1, colors1)
    add_cuboid(ax, vertices2, faces2, colors2)
    add_cuboid(ax, vertices3, faces3, colors3)

    # Set axes limits and view
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 5)
    ax.view_init(elev=35.7, azim=226)

    # Show the plot
    plt.show()

# Call the function to draw
draw_custom_shapes()