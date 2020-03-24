from unittest import TestSuite, TextTestRunner, defaultTestLoader

test_modules = [
    'databases.mongo_instance_test',
    'services.convert_service_test.py',
    'services.file_service_test'
]

suite = TestSuite()

for test_module in test_modules:
    try:
        module = __import__(test_module, globals(), locals(), ['suite'])
        suite_function = getattr(module, 'suite')
        suite.addTest(suite_function())
    except (ImportError, AttributeError):
        suite.addTest(defaultTestLoader.loadTestsFromName(test_module))

TextTestRunner().run(suite)
