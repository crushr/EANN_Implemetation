def parse_arguments(parser):
    '''
    parse_arguments(parser)函数：
        input   ：parser，一个argparse.ArgumentParser()对象
        function：通过parser.add_argument给parser作为属性不断添加参数
        return  ：包含所有参数的parser对象
    '''
    parser.add_argument('--output_file', type=str, default='/home/madm/Documents/EANN_recon/result/', help='')
    parser.add_argument('--best_validate_dir', type=str, default='/home/madm/Documents/EANN_recon/result/20.pkl', help='')
    parser.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--text_only', type=bool, default=False, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=20, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')

    return parser