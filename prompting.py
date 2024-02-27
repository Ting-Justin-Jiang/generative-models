PROMPT = [
    "A futuristic cityscape at dusk with neon lights reflecting in the water.",
    "A surreal landscape with floating islands and waterfalls in the sky.",
    "An ancient library filled with magical books and glowing orbs.",
    "A cyberpunk street market bustling with robots and humans.",
    "A serene garden with a path leading to a hidden cottage.",
    "An underwater city with coral skyscrapers and fish swimming in the air.",
    "A space station orbiting a vibrant alien planet.",
    "A post-apocalyptic world where nature has reclaimed a deserted city.",
    "A mystical forest with enchanted creatures and a glowing river.",
    "A steampunk workshop with intricate machinery and steam pipes.",
    "An ice castle on a snowy mountain lit by the aurora borealis.",
    "A desert oasis at night under a star-filled sky.",
    "A medieval village preparing for a festival with banners and lanterns.",
    "A giant's kitchen with oversized furniture and food.",
    "A magical academy with students practicing spells and flying on brooms.",
    "A treasure chamber inside an ancient pyramid, filled with gold and jewels.",
    "A superhero city at night, illuminated by a signal in the sky.",
    "A haunted mansion with ghosts and hidden passages.",
    "An alien bazaar with stalls selling exotic goods and creatures.",
    "A dreamy beach with bioluminescent waves and a starry sky.",
    "A futuristic laboratory with advanced technology and holograms.",
    "A wild west town with duels and horse-drawn carriages.",
    "A Viking feast in a great hall with a roaring fire.",
    "A carnival in Venice with elaborate masks and gondolas.",
    "A robot uprising in a city, with humans and robots in a standoff.",
    "An enchanted castle surrounded by a thorn forest.",
    "A pirate ship sailing through stormy seas with a rainbow in the distance.",
    "A mountain temple with monks meditating amidst cherry blossoms.",
    "An art nouveau city with intricate architecture and flowing designs.",
    "A fantasy marketplace with magical potions and mythical pets for sale.",
    "A dinosaur sanctuary with various species roaming in a lush valley.",
    "A cybernetic zoo with robotic animals and interactive exhibits."
]

prompts_file_path = "prompts.txt"
with open(prompts_file_path, "w") as file:
    for prompt in PROMPT:
        file.write(prompt + "\n")