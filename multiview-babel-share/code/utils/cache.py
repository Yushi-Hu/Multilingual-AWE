import logging as log
import os
import atexit
import signal
import shutil
import getpass


class Cache:

    def __init__(self):

        self.cache = []
        self.cachedir = f"/share/data/speech/Data/yushihu/scratch/{getpass.getuser()}/{os.getpid()}/"
        os.makedirs(self.cachedir, exist_ok=True)

        atexit.register(self.delete_cache)
        signal.signal(signal.SIGINT, self.delete_cache)
        signal.signal(signal.SIGTERM, self.delete_cache)
        log.info("Cache to delete on exit.")

    def __call__(self, src):
        # need the last folder to distinguish languages
        path, fn = os.path.split(src)
        dst = os.path.join(self.cachedir, os.path.basename(path)+fn)
        if dst not in self.cache:
            shutil.copyfile(src, dst)
            log.info(f"Caching at {dst}")
            self.cache.append(dst)

        return dst

    def delete_cache(self):

        for dst in self.cache:
            if os.path.exists(dst):
                os.remove(dst)
            if os.path.exists(dst):
                log.info(f"Delete failed: {dst}.")
            log.info(f"Deleted {dst}.".format(dst))

        if os.path.exists(self.cachedir):
            os.rmdir(self.cachedir)
        if os.path.exists(self.cachedir):
            log.info(f"Delete failed: {self.cachedir}.")
        log.info(f"Deleted {self.cachedir}.")
