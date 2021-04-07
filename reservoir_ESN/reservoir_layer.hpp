#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

#define LINEAR (0)
#define NON_LINEAR (1)

class reservoir_layer {
public:
	unsigned int unit_size;										//�@�m�[�h��
	unsigned int connection_degree;								//	1���j�b�g������̐ڑ���	(�m�[�h����1�����x�Ő��x���O�a����j
	double input_signal_strength;						//	���͂̋���
	double weight_factor;								//	���j�b�g�Ԑڑ��̋���
	std::vector<std::vector<double>> weight_reservoir;	//  ���U�[�o�[�w�̌����d��
	std::vector<std::vector<int>> adjacency_list;		//  �O���t�ɂ�����אڃ��X�g(�אڃ��X�g�H��https://qiita.com/drken/items/4a7869c5e304883f539b)
	unsigned int seed;									//	���U�[�o�[�̍\�������肷��V�[�h�l�i�\���̃V�[�h�Əd�݂̃V�[�h�Ȃǂ̕��������Ă����������j
	

};