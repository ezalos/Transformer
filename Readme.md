# My project

## Instalaltion

Just `git clone` me

## Do not forget

```sh
PROJECT="`printf '.venv_%s\n' "${PWD##*/}"`" &&  python -m venv $PROJECT && echo "conda deactivate\nsource $PROJECT/bin/activate" > .env && cd .
```

Alternative

```sh
PROJECT="`printf '.venv_%s\n' "${PWD##*/}"`" &&  python -m venv $PROJECT && echo "source $PROJECT/bin/activate" > .autoenv.zsh && cd .
```

## Usage

Just start writing your code
