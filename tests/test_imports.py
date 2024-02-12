#!/usr/bin/env python3
import warnings

def test_acgc():
    import acgc 

def test_erroranalysis():
    import acgc.erroranalysis

def test_figstyle():
    import acgc.figstyle

def test_gc():
    import acgc.gc

def test_hysplit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        import acgc.hysplit

def test_icartt():
    import acgc.icartt

def test_igra():
    import acgc.igra

def test_mapping():
    import acgc.mapping

def test_met():
    import acgc.met

def test_modetools():
    import acgc.modetools

def test_netcdf():
    import acgc.netcdf

def test_solar():
    import acgc.solar

def test_stats():
    import acgc.stats

def test_time():
    import acgc.time
