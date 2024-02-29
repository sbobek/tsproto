.. tsproto documentation master file, created by
   sphinx-quickstart on Fri Feb 23 10:20:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the TSProto documentation
---------------------------------



**TSProto (Post-host prototype-based explanations with rules for time-series classifiers)** is an XAI algorithm that produces explanations for any type of machine-learning model.
It provides local explanations in a form of human-readable (and executable) rules, but also provide counterfactual explanations as well as visualization of the explanations.

Install
=======

LUX can be installed from either `PyPI <https://pypi.org/project/tsproto>`_ or directly from source code `GitHub <https://github.com/sbobek/tsproto>`_

To install form PyPI::

   pip install tsproto

To install from source code::

   git clone https://github.com/sbobek/tsproto
   cd tsproto
   pip install .

.. toctree::
   :maxdepth: 2
   :caption: Examples

   Basic Usage examples <basic_examples>


.. toctree::
   :maxdepth: 2
   :caption: Reference

   API reference <api>

.. toctree::
   :maxdepth: 1
   :caption: Development

   Release notes <release_notes>
   Contributing guide <contributing>


