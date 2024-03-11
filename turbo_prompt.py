PROMPT = [
    "A cat that is sitting on a table on mars.",
    "Product shot of nike shoes, with soft vibrant colors, 3d blender render, modular constructivism, blue background, physically based rendering, centered",
    "An aerial view of a city at  night, long exposure, instagram contest",
    "Portrait photo of a storm trooper with his  beautiful wife on his wedding day",
    "Darth Vader at a convenience store, pushing shopping cart, CCTV still, high-angle  security camera feed",
    "Night club, people dancing, Fish-eye lens",
    "Fallout concept art school interior render grim, sun rays coming through window, unreal engine 5",
    "water color painting of sunset behind mountains, detailed, vaporwave aesthetic.",
    "Portrait by Derek Gores",
    "Horror by tim burton",
    "Art by Katsuhiro Otomo",
    "Oil painting of a man staring at the stars, by Frank Frazetta",
    "Tiny cute isometric kitchen in a cutaway box, soft smooth lighting, soft colors, 100mm lens, 3d blender render",
    "Anatomy of Pikachu, dissection Scientific illustration from a biology book",
    "Vector illustration of Living Room in Flat Style, pastel color palette",
    "Peppa pig, in Ukiyo-e style"
]

prompts_file_path = "prompts.txt"
with open(prompts_file_path, "w") as file:
    for prompt in PROMPT:
        file.write(prompt + "\n")