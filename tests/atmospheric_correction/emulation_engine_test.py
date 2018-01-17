from multiply_atmospheric_corection.util.emulation_engine import AtmosphericEmulationEngine

PATH_TO_EMUS = './multiply_atmospheric_corection/emus'


def test_load_emulators():
    AEE = AtmosphericEmulationEngine('MSI', PATH_TO_EMUS)
    up_bounds = AEE.emulators[0].inputs[:, 4:7].max(axis=0)
    print(up_bounds)
