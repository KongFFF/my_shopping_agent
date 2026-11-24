"""Attach the tested exit guard to C800; never start or alter training."""
import watch_c500_shutdown as base


def main():
    original=base.bind_supervisor
    base.bind_supervisor=lambda run:original(run,experiment='C800',supervisor_script='c800_run.py')
    try:base.main()
    finally:base.bind_supervisor=original


if __name__=='__main__':main()
