class Parser :
    def __init__(self, args):
        if args.parser_type == 'pascal_voc':
            from parser.pascal_voc_parser import get_data
        elif args.parser_type == 'simple':
            from parser.simple_parser import get_data
        else:
            raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")
        self.get_data = get_data
