//
//  ExploreViewController.swift
//  luminate
//
//  Created by Prayash Thapa on 10/7/17.
//  Copyright © 2017 com.reality.af. All rights reserved.
//

import LBTAComponents

class ExploreViewController: DatasourceController {
    override func viewDidLoad() {
        super.viewDidLoad()
        setupNavigationBarItems()
        self.title = "Explore a Cause"
        
        collectionView?.backgroundColor = UIColor(r: 255, g: 255, b: 255)
        
        self.datasource = HomeDatasource()
        
        // Hack to push AR scene right away
        let vc = ARViewController()
        self.navigationController?.pushViewController(vc, animated: true)
    }
    
    func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout, minimumLineSpacingForSectionAt section: Int) -> CGFloat {
        return 0
    }

    
    override func willTransition(to newCollection: UITraitCollection, with coordinator: UIViewControllerTransitionCoordinator) {
        collectionViewLayout.invalidateLayout()
    }
    
    override func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        let vc = ARViewController()
        self.navigationController?.pushViewController(vc, animated: true)
    }
    
    override func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout, sizeForItemAt indexPath: IndexPath) -> CGSize {
        return CGSize(width: view.frame.width, height: 200)
    }
}
