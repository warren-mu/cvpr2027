import unittest
from src.envs.pair_v2_construct import identity_sets, view_proposals, placement_groups, construction_pairs, adjacent_moves


def obj(name, category, x):
    return {'name': name, 'objectType': category, 'pickupable': True,
            'position': {'x': x, 'y': 1., 'z': 0.}}


class ConstructTests(unittest.TestCase):
    def test_unique_category_instruction(self):
        a, b, x = obj('a', 'Cup', 0), obj('b', 'Cup', 1), obj('x', 'Apple', 0)
        self.assertEqual(len(identity_sets([a,b,x])), 1)
        self.assertEqual(identity_sets([a,b,x,obj('c','Cup',2)]), [])
        self.assertEqual(identity_sets([a,b,x,obj('y','Apple',2)]), [])

    def test_order_independent_views_and_45_headings(self):
        a, b = obj('a','Cup',0), obj('b','Cup',1)
        points = [{'x':0.,'y':0.,'z':-1.},{'x':1.,'y':0.,'z':-1.}]
        self.assertEqual(view_proposals(points,a,b), view_proposals(points[::-1],a,b))
        self.assertTrue(all(r['yaw'] % 45 == 0 for r in view_proposals(points,a,b)))

    def test_ratio_does_not_invent_placement_success(self):
        a, b, x = obj('a','Cup',0), obj('b','Cup',1), obj('x','Apple',0)
        tokens = [{'position': {'x':p,'y':1.,'z':.1}, 'receptacle_name':'table'} for p in (.1,.2,.5,.8,.9)]
        left, right = placement_groups(tokens,a,b,x)
        self.assertEqual(len(left),2)
        self.assertEqual(len(right),2)
        plans = list(construction_pairs(left,right))
        self.assertEqual([p['mode'] for p in plans[:2]], ['flip','stay'])
        self.assertTrue(all('passed' not in p for p in plans))
        self.assertEqual(len(plans),6)

    def test_adjacency_excludes_diagonal_and_self(self):
        start = {'x':0.,'y':0.,'z':0.}
        points = [start,{'x':.25,'y':0.,'z':0.},{'x':.1,'y':0.,'z':.1},{'x':1.,'y':0.,'z':0.}]
        self.assertEqual(adjacent_moves(points,start), [points[1]])


if __name__ == '__main__':
    unittest.main()
