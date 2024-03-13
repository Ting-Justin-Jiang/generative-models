PROMPT = [
    "A cat is sitting on a table on mars",
    "Product shot of a pair of black Nike Air Jordan 11 shoes",
    "Art by Shinkai Makoto in the style of Your Name",
    "Fallout concept art school interior render grim, sun rays coming through window",
]

prompts_file_path = "prompts.txt"
with open(prompts_file_path, "w") as file:
    for prompt in PROMPT:
        file.write(prompt + "\n")