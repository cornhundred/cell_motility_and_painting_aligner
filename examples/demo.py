from cell_motility_painting_aligner import MotilityPaintingAligner

# Replace image URLs, sizes, and centroids with experiment data.
w = MotilityPaintingAligner.from_urls(
    motility_image_url="motility.png",
    cell_painting_image_urls=["cell_painting_001.png", "cell_painting_002.png"],
    motility_size=[512, 512],
    cell_painting_sizes=[[2048, 2048], [2048, 2048]],
    motility_centroids=[
        {"id": "mot_1", "x": 120, "y": 90},
        {"id": "mot_2", "x": 180, "y": 145},
        {"id": "mot_3", "x": 250, "y": 210},
    ],
    cell_painting_centroids_by_image=[
        [
            {"id": "cp_1", "x": 1010, "y": 820},
            {"id": "cp_2", "x": 1210, "y": 995},
            {"id": "cp_3", "x": 1475, "y": 1210},
        ],
        [
            {"id": "cp_1", "x": 890, "y": 760},
            {"id": "cp_2", "x": 1090, "y": 930},
            {"id": "cp_3", "x": 1350, "y": 1145},
        ],
    ],
)

w

