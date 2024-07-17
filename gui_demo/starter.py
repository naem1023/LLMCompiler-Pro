import chainlit as cl

demo_starter = [
    cl.Starter(
        label="Start Chat about US stock",
        message="Can you find some of the newest and most promising stocks in the US?",
        icon="/public/starter.svg",
    ),
    cl.Starter(
        label="Topic of origin LLM Compiler(Market Gap)",
        message="How much does Microsoft's market cap need to increase to exceed Apple's market cap?",
        icon="/public/starter.svg",
    ),
    cl.Starter(
        label="Topic of origin LLM Compiler(Movie Search)",
        message="Find a movie similar to Mission Impossible, The Silence of the Lambs, American Beauty, Star Wars Episode IV - A New Hope Options: Austin Powers International Man of Mystery Alesha Popvich and Tugarin the Dragon In Cold Blood Rosetta",
        icon="/public/starter.svg",
    ),
]
