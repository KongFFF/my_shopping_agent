"""Attach the tested exit guard to C650; never start or alter training."""
import watch_c500_shutdown as base


def main():
    original=base.bind_supervisor
    base.bind_supervisor=lambda run:original(run,experiment='C650',supervisor_script='c650_run.py')
    try:base.main()
    finally:base.bind_supervisor=original


if __name__=='__main__':main()
