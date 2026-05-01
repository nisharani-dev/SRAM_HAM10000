# config_6T.py

# SRAM organization
word_size = 16          # bits per word
num_words = 128         # number of words
num_banks = 1           # single bank

# Technology
tech_name = "freepdk45"

# Output
output_path = "6T_simulation/macro_6T"
output_name = "sram_6T"


route_supplies = False
check_lvsdrc = False
analytical_delay = True