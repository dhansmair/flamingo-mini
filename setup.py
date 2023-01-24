from setuptools import setup, find_packages

setup(
  name = 'flamingo-mini',
  packages = find_packages(exclude=[]),
  version = '0.0.2',
  license='MIT',
  description = 'flamingo-mini',
  author = 'dhansmair',
  author_email = 'd.hansmair@campus.lmu.de',
  url = 'https://github.com/dhansmair/flamingo-mini',
  python_requires='>=3.7',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'image captioning'
  ],
  install_requires=[
    'einops>=0.4',
    'einops-exts',
    'torch>=1.6',
    'transformers>=4.25.1',
    'numpy',
    'pillow'
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    # 'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)
