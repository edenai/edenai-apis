
from edenai_apis import Text

#sentiment_analysis = Text.sentiment_analysis("tenstorrent")
#tt_res = sentiment_analysis(language="en", text="as simple as that")

keyword_extraction = Text.keyword_extraction("tenstorrent")
#tt_res = keyword_extraction(language="en", text="Water is an inorganic compound with the chemical formula H2O. It is a transparent, tasteless, odorless, and nearly colorless chemical substance, and it is the main constituent of Earths hydrosphere and the fluids of all known living organisms.")

tt_res = keyword_extraction(language="en", text="Water is an inorganic compound with the chemical formula H2O. It is a transparent, tasteless, odorless, and nearly colorless chemical substance, and it is the main constituent of Earths hydrosphere and the fluids of all known living organisms (in which it acts as a solvent). It is vital for all known forms of life, despite not providing food energy, or organic micronutrients. Its chemical formula, H2O, indicates that each of its molecules contains one oxygen and two hydrogen atoms, connected by covalent bonds. The hydrogen atoms are attached to the oxygen atom at an angle of 104.45°. Water is also the name of the liquid state of H2O at standard temperature and pressure. Because Earths environment is relatively close to waters triple point, water exists on Earth as a solid, liquid, and gas. It forms precipitation in the form of rain and aerosols in the form of fog. Clouds consist of suspended droplets of water and ice, its solid state. When finely divided, crystalline ice may precipitate in the form of snow. The gaseous state of water is steam or water vapor.")

# Provider's response
print(tt_res.original_response)

# Standardized version of Provider's response
print(tt_res.standardized_response)

