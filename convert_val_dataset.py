import json

annotation_file = "data/val/annotations.json"
with open(annotation_file) as f:
    annotations = json.load(f)

for image in annotations['annotations']:
    # Iterate over the keypoints
    for keypoint in image['keypoints']:
        # Change the third element to 0
        if keypoint[2] == 1:
            keypoint[2] = 0

# Save the modified data back to the JSON file
with open('keypoints.json', 'w') as f:
    json.dump(annotations, f, indent=4)