## shablona
A bare-bones template for Python packages, ready for use with setuptools (PyPI), pip, and py.test.

### Using this as a template
Let's assume that you want to create a small scientific Python project called `smallish`.

To use this repository as a template, click the green "use this template" button on the front page of the "shablona" repository.

In "Repository name" enter the name of your project. For example, enter `smallish` here. After that, you can hit the "Create repository from template" button.

You should then be able to clone the new repo into your machine. You will want to change the names of the files. For example, you will want to move `shablona/shablona.py` to be called `smallish/smallish.py`
```
git mv shablona smallish
git mv smallish/shablona.py smallish/smallish.py
git mv smallish/tests/test_shablona.py smallish/tests/test_smallish.py
```

Make a commit recording these changes. Something like:
```
git commit -a -m "Moved names from `shablona` to `smallish`"
```

You will want to edit a few more places that still have `shablona` in them. Type the following to see where all these files are:
```
git grep shablona
```

You can replace `shablona` for `smallish` quickly with:
```
git grep -l 'shablona' | xargs sed -i 's/shablona/smallish/g'
```

Edit `shablona/__init__.py`, and `shablona/version.py` with the information specific to your project.

This very file (`README.md`) should be edited to reflect what your project is about.

At this point, make another commit, and continue to develop your own code based on this template.


### Contributing
If you wish to make any changes (e.g. add documentation, tests, continuous integration, etc.), please follow the [Shablona](https://github.com/uwescience/shablona) template.