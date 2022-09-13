##############################################################################
MoeGoe Repacked
##############################################################################
| Repacked as a FastAPI application. `api.py <api.py>`_ is the entry point.
|
| The original README.md is `here <moegoe/README.md>`_.
|
| 1. Put your checkpoints and config files in the `statics <statics>`_ directory with specific names.
|
| 2. Add them to the ``models`` varieable (`here <api.py#L25-L29>`_) with the following format:
|
.. code:: python

 models = {
     "hamidashi": ["hamidashi_604_epochs.pth", "hamidashi_config.json"],
     "yuzusoft": ["yuzusoft_365_epochs.pth", "yuzusoft_config.json"],
 }
| 3. Create a subclass of ``BaseModel`` and its implementation using your models as `Hamidashi <api.py#L71-L115>`_ or `Yuzusoft <api.py#L118-L159>`_.
|
| 4. Put your ``Model.syn()`` function before or after the general ``syn()`` function (`here <api.py#L203-L204>`_).
|
| 5. Configure the listening ``host`` and ``port`` `here <api.py#L274>`_.
|
| 6. Run the application.
******************************************************************************
Usage
******************************************************************************
| The application will run the `sample <api.py#L220-L242>`_ for audition at startup.
|
| Use ``http://localhost:10721`` to access the application by default.
* \/text
   | ``text`` to synthesize, return the synthesized WAV file directly.
   | ``http://localhost:10721/让宁宁说こんにちは`` -> ``宁宁: こんにちは``

    The application will use the file that is already existing/synthesized in the `output <output>`_ directory as cache.
* /text?local
   | ``text`` to synthesize, return the **local absolute path** of synthesized WAV file.
   | ``http://localhost:10721/让芳乃说こんにちは?local`` -> ``./.../filename.wav``

    Avaliable only when accessed by localhost. Act as ``/text`` otherwise.
* /text?remote
   | ``text`` to synthesize, return the **remote path snippet** of synthesized WAV file.
   | ``http://localhost:10721/让妃爱说こんにちは?remote`` -> ``/output/filename.wav``
* /output/filename
   | ``filename`` that is already synthesized, return the synthesized WAV file directly.
   | Use ``/text?remote`` to get the remote path snippet.
   | ``http://localhost:10721/output/filename.wav`` -> ``妃爱: こんにちは``