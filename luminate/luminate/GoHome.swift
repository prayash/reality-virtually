//
//  GoHome.swift
//  luminate
//
//  Created by Jesse Litton on 10/8/17.
//  Copyright © 2017 com.reality.af. All rights reserved.
//

import UIKit

class GoHomeButtonView: UIView {
    var button = UIButton()
    
    override init (frame: CGRect) {
        super.init(frame: frame)
        self.backgroundColor = UIColor.clear
        self.translatesAutoresizingMaskIntoConstraints = false

        setupButton()
    }
    
    convenience init() {
        self.init(frame: CGRect.zero)
    }
    
    func setupButton() {
        let button: UIButton = {
            let view = UIButton()
            view.setTitle("HOME", for: .normal)
            view.setTitleColor(UIColor.white, for: .normal)
            view.backgroundColor = UIColor(red: 1, green: 1, blue: 1, alpha: 0.25)
            return view
        }()
        
        button.addTarget(self, action: #selector(hideSelf), for: .touchUpInside)
        
        addSubview(button)
        button.anchor(nil, left: self.leftAnchor, bottom: self.bottomAnchor, right: self.rightAnchor, topConstant: 0, leftConstant: 12, bottomConstant: 40
            , rightConstant: 12, widthConstant: 18, heightConstant: 55)
        button.layer.cornerRadius = 28
    }
    
    @objc func hideSelf() {
        self.isHidden = true
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

