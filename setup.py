import os
import re
import ast
import shutil
from itertools import product
from setuptools import setup, find_packages


def get_version(file_name, version_name="__version__"):
    with open(file_name) as f:
        tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if node.targets[0].id == version_name:
                    return node.value.s
    raise ValueError(f"Unable to find an assignment to the variable {version_name} in file {file_name}")


class About(object):
    NAME = 'audio8'
    AUTHOR = 'dpressel'
    VERSION = get_version('audio8/version.py')
    EMAIL = "{}@gmail.com".format(AUTHOR)
    URL = f"https://www.github.com/mead-ml/{NAME}"
    DOWNLOAD_URL = "{}/archive/{}.tar.gz".format(URL, VERSION)
    DOC_URL = "{}/tree/master/".format(URL)
    DOC_NAME = 'docs/{}.md'.format(NAME)


def write_manifest(lines):
    with open("MANIFEST.in", "w") as f:
        f.write("\n".join(map(lambda x: 'include {}'.format(x), lines)))


def fix_links(text):
    """Pypi doesn't seem to host multiple docs so replace local links with ones to github."""
    regex = re.compile(r"\[(.*?)\]\(((?:docs|docker)/.*?\.md)\)")
    text = regex.sub(r"[\1]({}\2)".format(About.DOC_URL), text)
    return text


def read_doc(f_name, new_name=None, fix_fn=fix_links):
    """
    Because our readme is outside of this dir we need to copy it in so
    that it is picked up by the install.
    """
    if new_name is None:
        new_name = f_name
    path = os.path.dirname(os.path.realpath(__file__))
    doc_loc = os.path.normpath(os.path.join(path, '..', f_name))
    new_loc = os.path.join(path, new_name)
    if os.path.isfile(doc_loc):
        shutil.copyfile(doc_loc, new_loc)
    descript = open(new_loc, 'r').read()
    return fix_fn(descript)


def main():
    setup(
        name='mead-{}'.format(About.NAME),
        version=About.VERSION,
        description='MEAD Audio',
        long_description=read_doc(About.DOC_NAME, new_name='README.md'),
        long_description_content_type="text/markdown",
        author=About.AUTHOR,
        author_email=About.EMAIL,
        license='Apache 2.0',
        url=About.URL,
        download_url=About.DOWNLOAD_URL,
        packages=find_packages(exclude=['tests', 'api-examples']),
        package_data={},
        include_package_data=False,
        install_requires=['numpy', 'six', 'soundfile', 'editdistance', 'mead-baseline',],
        extras_require={
            'test': ['pytest', 'mock', 'contextdecorator', 'pytest-forked'],
            'scipy': ['scipy'],
            'ctcdecode': ['ctcdecode'],
        },
        entry_points={'console_scripts': []},
        classifiers={
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        },
        keywords=['deep-learning', 'audio', 'pytorch'],
    )


if __name__ == "__main__":
    main()
