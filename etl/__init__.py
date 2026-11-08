import sys

# Consola Windows (cp1252) nu poate afișa diacriticele românești; nu lăsăm print() să oprească scripturile.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except Exception:
        pass
