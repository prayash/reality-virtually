//
//  ExploreViewController.swift
//  luminate
//
//  Created by Prayash Thapa on 10/8/17.
//  Copyright © 2017 com.reality.af. All rights reserved.
//

import Foundation

import UIKit

class ExploreViewController: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if let vc = segue.destination as? ARViewController {
            vc.isGiving = false
        }
    }
}
