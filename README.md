# LLMCompiler-Pro
_An extended project of the LLM Compiler paper, focusing on developing LLM-based Autonomous Agents._

This project is an extension of the LLMCompiler research, developed with a focus on enhancing conversational capabilities, expanding the flexibility of plan decomposition, and broadening the concept of tools.

## Objective
- Enable integration with a wide range of APIs and tools beyond simple Python functions.
- Provide an interactive conversational interface. 
- Handle subtasks generated through plan decomposition with function calling.
- Maintain the fast processing speed of the original LLMCompiler.
- Ensure users do not experience waiting by offering various streaming interactions.
- Comprehensive refactoring and migration of deprecated modules from langchain and OpenAI.

## Architecture
![](assets/llmcompilerpro.png)

## Dependencies Installation
- Prepare the python 3.12 using conda, pyenv, or any other method.
- Prepare the docker for docker-compose.

```shell
pip install poetry
poetry install --with dev --no-roots
make up
```
## Run Demo
```shell
chainlit run main.py
````

## Contribution Guidelines
Please make sure to use pre-commit hooks to ensure code quality and consistency.
```shell
pre-commit install -c .conf/.pre-commit.yaml
```

# Citations
```
@misc{kim2024llmcompilerparallelfunction,
      title={An LLM Compiler for Parallel Function Calling}, 
      author={Sehoon Kim and Suhong Moon and Ryan Tabrizi and Nicholas Lee and Michael W. Mahoney and Kurt Keutzer and Amir Gholami},
      year={2024},
      eprint={2312.04511},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2312.04511}, 
}
```
